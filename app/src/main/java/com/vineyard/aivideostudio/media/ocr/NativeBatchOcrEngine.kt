package com.vineyard.aivideostudio.media.ocr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.vineyard.aivideostudio.media.tools.OcrLineData
import com.vineyard.aivideostudio.media.tools.OcrWordData
import com.vineyard.aivideostudio.media.tools.ToolsBoundingBox
import com.vineyard.aivideostudio.media.video.ExtractedFrame
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.isActive
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.min

/**
 * Structured OCR result representing all detected text blocks, lines, and words on a single frame.
 */
data class FrameOcrData(
    val frameIndex: Int,
    val time: Float,
    val lines: List<OcrLineData>
) {
    /**
     * Serializes this frame's OCR bounding boxes into standard JSON format
     * matching the exact schema expected by the timeline exporter and ZIP parser.
     */
    fun toJsonString(): String {
        val root = JSONObject()
        root.put("frame", frameIndex)
        val formattedTime = String.format(Locale.US, "%.2f", time).toDoubleOrNull() ?: time.toDouble()
        root.put("time", formattedTime)

        val linesArray = JSONArray()
        for (line in lines) {
            val lineObj = JSONObject()
            lineObj.put("text", line.text)

            val bboxObj = JSONObject()
            bboxObj.put("x0", line.bbox.x0.toInt())
            bboxObj.put("y0", line.bbox.y0.toInt())
            bboxObj.put("width", line.bbox.width.toInt())
            bboxObj.put("height", line.bbox.height.toInt())
            lineObj.put("bbox", bboxObj)

            val wordsArray = JSONArray()
            for (word in line.words) {
                val wordObj = JSONObject()
                wordObj.put("text", word.text)

                val wordBboxObj = JSONObject()
                wordBboxObj.put("x0", word.bbox.x0.toInt())
                wordBboxObj.put("y0", word.bbox.y0.toInt())
                wordBboxObj.put("width", word.bbox.width.toInt())
                wordBboxObj.put("height", word.bbox.height.toInt())
                wordObj.put("bbox", wordBboxObj)

                wordsArray.put(wordObj)
            }
            lineObj.put("words", wordsArray)
            linesArray.put(lineObj)
        }
        root.put("lines", linesArray)
        return root.toString(2)
    }
}

/**
 * High-speed native OCR engine using on-device Google ML Kit Vision.
 * Features perceptual frame deduplication and efficient memory recycling.
 */
class NativeBatchOcrEngine {

    private val textRecognizer: TextRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    /**
     * Scans a single frame from its disk cache path and extracts lines and word bounding boxes.
     */
    suspend fun scanFrame(frame: ExtractedFrame): FrameOcrData = withContext(Dispatchers.IO) {
        val file = File(frame.fullResImagePath)
        if (!file.exists()) {
            return@withContext FrameOcrData(frame.index, frame.timeSeconds, emptyList())
        }

        val decodeOptions = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = BitmapFactory.decodeFile(file.absolutePath, decodeOptions)
            ?: return@withContext FrameOcrData(frame.index, frame.timeSeconds, emptyList())

        return@withContext try {
            val inputImage = InputImage.fromBitmap(bitmap, 0)
            val visionText: Text = textRecognizer.process(inputImage).await()
            val extractedLines = parseVisionText(visionText)

            FrameOcrData(
                frameIndex = frame.index,
                time = frame.timeSeconds,
                lines = extractedLines
            )
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            e.printStackTrace()
            FrameOcrData(frame.index, frame.timeSeconds, emptyList())
        } finally {
            bitmap.recycle() // Keep RAM footprint clean
        }
    }

    /**
     * Scans multiple frames at high speed with perceptual deduplication and IDM-style parallel concurrency.
     * Divides workload across concurrent worker streams and processes sub-batches of 10 frames simultaneously.
     */
    suspend fun scanBatch(
        frames: List<ExtractedFrame>,
        parallelWorkers: Int = 5,
        onProgress: (current: Int, total: Int) -> Unit
    ): Map<Int, FrameOcrData> = withContext(Dispatchers.IO) {
        val totalCount = frames.size
        if (totalCount == 0) return@withContext emptyMap()

        // Sort chronologically before splitting to preserve timeline sequence
        val sortedFrames = frames.sortedBy { it.index }
        val resultsMap = ConcurrentHashMap<Int, FrameOcrData>()
        val completedCounter = AtomicInteger(0)
        val lastProgressDispatchTime = AtomicLong(0L)

        // IDM partition into 5 concurrent worker streams
        val workerCount = min(parallelWorkers.coerceAtLeast(1), 5)
        val chunkSize = ceil(totalCount.toFloat() / workerCount.toFloat()).toInt()

        val deferredWorkers = (0 until workerCount).map { workerIndex ->
            val startIdx = workerIndex * chunkSize
            val endIdx = min(totalCount, startIdx + chunkSize)

            async(Dispatchers.IO) {
                if (startIdx >= endIdx) return@async

                var lastScannedFrame: ExtractedFrame? = null
                var lastOcrData: FrameOcrData? = null

                // Process in mini-batches of 10
                val batchSize = 10
                var batchStart = startIdx

                while (batchStart < endIdx && isActive) {
                    val batchEnd = min(endIdx, batchStart + batchSize)

                    for (i in batchStart until batchEnd) {
                        if (!isActive) break
                        val frame = sortedFrames[i]

                        // Check perceptual similarity within this continuous timeline segment
                        if (lastScannedFrame != null && lastOcrData != null &&
                            areBitmapsSimilar(lastScannedFrame.thumbBitmap, frame.thumbBitmap)
                        ) {
                            val reusedData = FrameOcrData(
                                frameIndex = frame.index,
                                time = frame.timeSeconds,
                                lines = lastOcrData.lines
                            )
                            resultsMap[frame.index] = reusedData
                        } else {
                            val ocrData = scanFrame(frame)
                            resultsMap[frame.index] = ocrData
                            lastScannedFrame = frame
                            lastOcrData = ocrData
                        }

                        val completed = completedCounter.incrementAndGet()
                        val now = System.currentTimeMillis()
                        val lastTime = lastProgressDispatchTime.get()

                        if (now - lastTime > 60 || completed == totalCount) {
                            if (lastProgressDispatchTime.compareAndSet(lastTime, now)) {
                                withContext(Dispatchers.Main) {
                                    onProgress(completed, totalCount)
                                }
                            }
                        }
                    }

                    batchStart = batchEnd
                }
            }
        }

        deferredWorkers.awaitAll()

        withContext(Dispatchers.Main) {
            onProgress(resultsMap.size, totalCount)
        }

        resultsMap.toSortedMap()
    }

    /**
     * Fast sub-millisecond perceptual difference check using the in-memory thumbnails.
     * Returns true if differences between [bmpA] and [bmpB] are within [thresholdPercent].
     */
    private fun areBitmapsSimilar(bmpA: Bitmap?, bmpB: Bitmap?, thresholdPercent: Float = 2.0f): Boolean {
        if (bmpA == null || bmpB == null) return false
        if (bmpA.isRecycled || bmpB.isRecycled) return false
        if (bmpA.width != bmpB.width || bmpA.height != bmpB.height) return false

        val w = bmpA.width
        val h = bmpA.height
        val step = 4 // Sample 1 in every 16 pixels for microsecond execution
        var diffCount = 0
        var sampledCount = 0

        for (y in 0 until h step step) {
            for (x in 0 until w step step) {
                val pixelA = bmpA.getPixel(x, y)
                val pixelB = bmpB.getPixel(x, y)
                sampledCount++

                if (pixelA != pixelB) {
                    val rDiff = abs((pixelA shr 16 and 0xFF) - (pixelB shr 16 and 0xFF))
                    val gDiff = abs((pixelA shr 8 and 0xFF) - (pixelB shr 8 and 0xFF))
                    val bDiff = abs((pixelA and 0xFF) - (pixelB and 0xFF))
                    if (rDiff + gDiff + bDiff > 40) { // Color distance threshold (filters compression noise)
                        diffCount++
                    }
                }
            }
        }

        if (sampledCount == 0) return false
        val diffRatio = (diffCount.toFloat() / sampledCount.toFloat()) * 100f
        return diffRatio <= thresholdPercent
    }

    /**
     * Converts ML Kit Vision Text structure into normalized [OcrLineData] and [OcrWordData].
     */
    private fun parseVisionText(visionText: Text): List<OcrLineData> {
        val linesList = mutableListOf<OcrLineData>()

        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                val lineText = line.text.trim().replace(Regex("\\s+"), " ")
                if (lineText.isEmpty()) continue

                val lineRect = line.boundingBox ?: Rect(0, 0, 0, 0)
                val lineBoundingBox = ToolsBoundingBox(
                    x0 = lineRect.left.toFloat(),
                    y0 = lineRect.top.toFloat(),
                    width = lineRect.width().toFloat(),
                    height = lineRect.height().toFloat()
                )

                val wordsList = mutableListOf<OcrWordData>()
                for (element in line.elements) {
                    val wordText = element.text.trim()
                    if (wordText.isEmpty()) continue

                    val elementRect = element.boundingBox ?: Rect(0, 0, 0, 0)
                    val wordBoundingBox = ToolsBoundingBox(
                        x0 = elementRect.left.toFloat(),
                        y0 = elementRect.top.toFloat(),
                        width = elementRect.width().toFloat(),
                        height = elementRect.height().toFloat()
                    )

                    wordsList.add(
                        OcrWordData(
                            text = wordText,
                            bbox = wordBoundingBox
                        )
                    )
                }

                linesList.add(
                    OcrLineData(
                        text = lineText,
                        bbox = lineBoundingBox,
                        words = wordsList
                    )
                )
            }
        }

        return linesList
    }

    /**
     * Releases ML Kit native resources when shutting down the OCR subsystem.
     */
    fun close() {
        try {
            textRecognizer.close()
        } catch (_: Exception) {
        }
    }
}