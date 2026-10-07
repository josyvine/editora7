package com.vineyard.aivideostudio.media.video

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.ParcelFileDescriptor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Represents a single extracted frame ready for UI display, OCR scanning, and overlay rendering.
 */
data class ExtractedFrame(
    val index: Int,
    val timeSeconds: Float,
    val timeFormatted: String,
    val fullResImagePath: String,   // Path to the cached high-resolution image on disk
    val thumbBitmap: Bitmap,        // Lightweight in-memory bitmap for the filmstrip UI
    var isHighlightEnabled: Boolean = false,
    var hasMismatch: Boolean = false
)

/**
 * Metadata initialization parameters for one of the concurrent IDM worker tasks.
 */
data class WorkerInitInfo(
    val taskId: Int,
    val startFrame: Int,
    val endFrame: Int,
    val totalFrames: Int
)

/**
 * High-performance, multi-threaded native frame extraction engine.
 * Employs 10 isolated kernel file descriptors for true non-blocking parallel hardware decoding.
 */
class FastNativeFrameExtractor(private val context: Context) {

    /**
     * IDM-Style 10-Stream Parallel Extraction with per-worker real-time progress callbacks.
     * Each of the 10 concurrent streams decodes 10 frames per batch independently.
     */
    suspend fun extractFramesWithWorkers(
        videoUri: Uri,
        targetFps: Int,
        onInitWorkers: (List<WorkerInitInfo>) -> Unit,
        onWorkerProgress: (taskId: Int, doneInWorker: Int, totalInWorker: Int) -> Unit,
        onTotalProgress: (current: Int, total: Int) -> Unit
    ): List<ExtractedFrame> = withContext(Dispatchers.IO) {

        val workspaceDir = File(context.filesDir, "editora_frames_workspace")
        if (workspaceDir.exists()) {
            workspaceDir.deleteRecursively()
        }
        workspaceDir.mkdirs()

        val probeRetriever = MediaMetadataRetriever()
        var probePfd: ParcelFileDescriptor? = null
        val durationSec: Float
        val targetDecodeWidth: Int
        val targetDecodeHeight: Int

        try {
            probePfd = try {
                context.contentResolver.openFileDescriptor(videoUri, "r")
            } catch (_: Exception) {
                null
            }

            if (probePfd != null) {
                probeRetriever.setDataSource(probePfd.fileDescriptor)
            } else {
                probeRetriever.setDataSource(context, videoUri)
            }

            val durationMs = probeRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            durationSec = durationMs / 1000f

            val rawWidth = probeRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val rawHeight = probeRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val rotation = probeRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0

            val isRotated = rotation == 90 || rotation == 270
            val origWidth = if (isRotated) rawHeight else rawWidth
            val origHeight = if (isRotated) rawWidth else rawHeight

            val maxAllowedDimension = 1280
            var decW = rawWidth
            var decH = rawHeight

            if (origWidth > 0 && origHeight > 0) {
                if (origWidth >= origHeight && origWidth > maxAllowedDimension) {
                    val scale = maxAllowedDimension.toFloat() / origWidth
                    decW = if (isRotated) (rawWidth * scale).roundToInt() else maxAllowedDimension
                    decH = if (isRotated) maxAllowedDimension else (rawHeight * scale).roundToInt()
                } else if (origHeight > origWidth && origHeight > maxAllowedDimension) {
                    val scale = maxAllowedDimension.toFloat() / origHeight
                    decW = if (isRotated) maxAllowedDimension else (rawWidth * scale).roundToInt()
                    decH = if (isRotated) (rawHeight * scale).roundToInt() else maxAllowedDimension
                }
            }
            targetDecodeWidth = decW
            targetDecodeHeight = decH
        } finally {
            try { probeRetriever.release() } catch (_: Exception) {}
            try { probePfd?.close() } catch (_: Exception) {}
        }

        val totalFrames = floor(durationSec * targetFps).toInt()
        if (totalFrames <= 0) return@withContext emptyList()

        val intervalUs = (1_000_000L / targetFps)

        // IDM partition into 10 concurrent worker streams
        val workerCount = 10
        val chunkSize = ceil(totalFrames.toFloat() / workerCount.toFloat()).toInt()

        val workerInitList = (0 until workerCount).map { i ->
            val start = i * chunkSize
            val end = min(totalFrames - 1, start + chunkSize - 1)
            val framesInWorker = max(0, end - start + 1)
            WorkerInitInfo(
                taskId = i + 1,
                startFrame = start,
                endFrame = end,
                totalFrames = framesInWorker
            )
        }

        withContext(Dispatchers.Main) {
            onInitWorkers(workerInitList)
        }

        val completedCounter = AtomicInteger(0)

        val deferredWorkers = workerInitList.map { workerInfo ->
            async(Dispatchers.IO) {
                if (workerInfo.totalFrames <= 0) return@async emptyList<ExtractedFrame>()

                val workerFrames = mutableListOf<ExtractedFrame>()
                val workerRetriever = MediaMetadataRetriever()
                var workerPfd: ParcelFileDescriptor? = null
                var workerDoneCount = 0

                try {
                    // Each worker opens its OWN independent ParcelFileDescriptor and file offset
                    workerPfd = try {
                        context.contentResolver.openFileDescriptor(videoUri, "r")
                    } catch (_: Exception) {
                        null
                    }

                    if (workerPfd != null) {
                        workerRetriever.setDataSource(workerPfd.fileDescriptor)
                    } else {
                        workerRetriever.setDataSource(context, videoUri)
                    }

                    val batchSize = 10
                    var currentBatchStart = workerInfo.startFrame
                    val workerEndBound = workerInfo.endFrame + 1

                    while (currentBatchStart < workerEndBound && isActive) {
                        val currentBatchEnd = min(workerEndBound, currentBatchStart + batchSize)
                        var framesProcessedInBatch = 0

                        for (i in currentBatchStart until currentBatchEnd) {
                            if (!isActive) break

                            val targetTimeUs = i * intervalUs
                            val timeSec = targetTimeUs / 1_000_000f

                            // Hardware-accelerated frame decode
                            val frameBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1 && targetDecodeWidth > 0 && targetDecodeHeight > 0) {
                                workerRetriever.getScaledFrameAtTime(
                                    targetTimeUs,
                                    MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                                    targetDecodeWidth,
                                    targetDecodeHeight
                                ) ?: workerRetriever.getScaledFrameAtTime(
                                    targetTimeUs,
                                    MediaMetadataRetriever.OPTION_CLOSEST,
                                    targetDecodeWidth,
                                    targetDecodeHeight
                                ) ?: workerRetriever.getFrameAtTime(targetTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                            } else {
                                workerRetriever.getFrameAtTime(targetTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                            } ?: continue

                            // 1. Filmstrip lightweight thumbnail (120px)
                            val thumbWidth = 120
                            val aspect = frameBitmap.height.toFloat() / frameBitmap.width.toFloat()
                            val thumbHeight = (thumbWidth * aspect).roundToInt().coerceAtLeast(1)
                            val thumbBitmap = Bitmap.createScaledBitmap(frameBitmap, thumbWidth, thumbHeight, false)

                            // 2. High-speed 64KB buffered stream write
                            val frameFile = File(workspaceDir, "frame_$i.jpg")
                            try {
                                BufferedOutputStream(FileOutputStream(frameFile), 65536).use { outStream ->
                                    frameBitmap.compress(Bitmap.CompressFormat.JPEG, 75, outStream)
                                }
                            } catch (writeErr: Exception) {
                                writeErr.printStackTrace()
                            }

                            frameBitmap.recycle()

                            val timeFormatted = String.format(Locale.US, "%.2f", timeSec)

                            workerFrames.add(
                                ExtractedFrame(
                                    index = i,
                                    timeSeconds = timeSec,
                                    timeFormatted = timeFormatted,
                                    fullResImagePath = frameFile.absolutePath,
                                    thumbBitmap = thumbBitmap
                                )
                            )

                            framesProcessedInBatch++
                        }

                        // Emit progress per 10-frame batch completed
                        if (framesProcessedInBatch > 0) {
                            workerDoneCount += framesProcessedInBatch
                            val completed = completedCounter.addAndGet(framesProcessedInBatch)

                            withContext(Dispatchers.Main) {
                                onWorkerProgress(workerInfo.taskId, workerDoneCount, workerInfo.totalFrames)
                                onTotalProgress(completed, totalFrames)
                            }
                        }

                        currentBatchStart = currentBatchEnd
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                } finally {
                    try { workerRetriever.release() } catch (_: Exception) {}
                    try { workerPfd?.close() } catch (_: Exception) {}
                }

                workerFrames
            }
        }

        val allChunks = deferredWorkers.awaitAll()
        val sortedFrames = allChunks.flatten().sortedBy { it.index }

        withContext(Dispatchers.Main) {
            onTotalProgress(sortedFrames.size, totalFrames)
        }

        return@withContext sortedFrames
    }

    /**
     * Backward-compatible frame extraction interface.
     */
    suspend fun extractFrames(
        videoUri: Uri,
        targetFps: Int,
        onProgress: (current: Int, total: Int) -> Unit
    ): List<ExtractedFrame> {
        return extractFramesWithWorkers(
            videoUri = videoUri,
            targetFps = targetFps,
            onInitWorkers = {},
            onWorkerProgress = { _, _, _ -> },
            onTotalProgress = onProgress
        )
    }

    /**
     * Cleans up the disk workspace. Should be called when a new video is loaded or the app is closed.
     */
    fun clearWorkspace() {
        try {
            val workspaceDir = File(context.filesDir, "editora_frames_workspace")
            if (workspaceDir.exists()) {
                workspaceDir.deleteRecursively()
            }
        } catch (_: Exception) {}
    }
}