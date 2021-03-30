/*
 * Copyright (c) 2021, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.io.gstwebcamcapture;

import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.DataBufferInt;
import java.awt.image.WritableRaster;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import org.freedesktop.gstreamer.Buffer;
import org.freedesktop.gstreamer.Element;
import org.freedesktop.gstreamer.Pad;
import org.freedesktop.gstreamer.PadProbeInfo;
import org.freedesktop.gstreamer.PadProbeReturn;

/**
 * Responsible for receiving images from GST and calling all image listeners
 * @author Devin Willis
 */
public class GStreamerImagePipeline implements Pad.PROBE, ImageProducer {

    private final Lock bufferLock = new ReentrantLock();
    private final BufferedImage image;
    private final int[] data;
    private Element identity;

    /**
     * Creates a new GStreamerImagePipeline
     * @param width Width of Camera
     * @param height Height of Camera
     * @param identity GStreamer Identity Element from Pipeline
     */
    public GStreamerImagePipeline(int width, int height, Element identity) {
        image = new BufferedImage(width, height, BufferedImage.TYPE_INT_BGR);
        data = ((DataBufferInt) (image.getRaster().getDataBuffer())).getData();
        this.identity = identity;
    }

    /**
     * Probe callback function that receives and process image buffer from GST
     * @param pad
     * @param info
     * @return 
     */
    @Override
    public PadProbeReturn probeCallback(Pad pad, PadProbeInfo info) {

        if (false) {
            return PadProbeReturn.OK;
        }

        if (!bufferLock.tryLock()) {
            System.out.println("Busy");
            return PadProbeReturn.OK;  //https://gstreamer.freedesktop.org/documentation/gstreamer/gstpad.html?gi-language=c
        }

        try {

            Buffer buffer = info.getBuffer();
            if (buffer.isWritable()) {
                IntBuffer ib = buffer.map(true).asIntBuffer();
                ib.get(data);
                process();
                ib.rewind();
                ib.put(data);
                buffer.unmap();
            }

        } finally {
            bufferLock.unlock();
        }

        return PadProbeReturn.OK;
    }

    /**
     * processes image and sends to all image listeners
     */
    private void process() {
        for (ImageListener imageListener : imageListeners) {
            ColorModel cm = image.getColorModel();
            boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
            WritableRaster raster = image.copyData(image.getRaster().createCompatibleWritableRaster());
            BufferedImage copy = new BufferedImage(cm, raster, isAlphaPremultiplied, null);
            imageListener.newImage(copy);
        }

    }

    ArrayList<ImageListener> imageListeners = new ArrayList<ImageListener>();

    /**
     * Adds image listener to ImagePipeline class
     * @param imageListener
     * @return 
     */
    @Override
    public ImageProducer addImageListener(ImageListener imageListener) {
        imageListeners.add(imageListener);
        return this;
    }
}