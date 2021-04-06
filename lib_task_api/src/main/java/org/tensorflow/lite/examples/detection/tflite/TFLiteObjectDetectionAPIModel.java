package org.tensorflow.lite.examples.detection.tflite;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Trace;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions;

public class TFLiteObjectDetectionAPIModel implements Detector {
  private static final String TAG = "TFLiteObjectDetectionAPIModelWithTaskApi";
  private static final int NUM_DETECTIONS = 10;
  private final MappedByteBuffer modelBuffer;
  private ObjectDetector objectDetector;
  private final ObjectDetectorOptions.Builder optionsBuilder;

  public static Detector create(
      final Context context,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    return new TFLiteObjectDetectionAPIModel(context, modelFilename);
  }

  private TFLiteObjectDetectionAPIModel(Context context, String modelFilename) throws IOException {
    modelBuffer = FileUtil.loadMappedFile(context, modelFilename);
    optionsBuilder = ObjectDetectorOptions.builder().setMaxResults(NUM_DETECTIONS);
    objectDetector = ObjectDetector.createFromBufferAndOptions(modelBuffer, optionsBuilder.build());
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    Trace.beginSection("recognizeImage");
    List<Detection> results = objectDetector.detect(TensorImage.fromBitmap(bitmap));

    final ArrayList<Recognition> recognitions = new ArrayList<>();
    int cnt = 0;
    for (Detection detection : results) {
      recognitions.add(
          new Recognition(
              "" + cnt++,
              detection.getCategories().get(0).getLabel(),
              detection.getCategories().get(0).getScore(),
              detection.getBoundingBox()));
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    if (objectDetector != null) {
      objectDetector.close();
    }
  }

  @Override
  public void setNumThreads(int numThreads) {
    if (objectDetector != null) {
      optionsBuilder.setNumThreads(numThreads);
      recreateDetector();
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    throw new UnsupportedOperationException(
        "Manipulating the hardware accelerators is not allowed in the Task"
            + " library currently. Only CPU is allowed.");
  }

  private void recreateDetector() {
    objectDetector.close();
    objectDetector = ObjectDetector.createFromBufferAndOptions(modelBuffer, optionsBuilder.build());
  }
}
