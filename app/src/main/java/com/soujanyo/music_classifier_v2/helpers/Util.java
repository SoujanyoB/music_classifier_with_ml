package com.soujanyo.music_classifier_v2.helpers;

import android.app.Activity;
import android.content.Context;
import android.util.Log;

import java.nio.ByteBuffer;

public class Util {

    /**
     * @param tag     tag for the log message
     * @param message the message to be displayed
     */
    public static void logInfo(String tag, String message) {
        Log.d(tag, message);
    }

    /**
     * @param array input array to find position of max element
     * @return position of max element
     */
    public static int getIndexOfMax(float[] array) {

        int index = 0;
        float max = Float.MIN_VALUE;

        for (int i = 0; i < array.length; i++) {

            if (array[i] > max) {
                max = array[i];
                index = i;
            }

        }

        return index;
    }


    /**
     * @param array input array of mfcc
     * @return data buffer for the corresponding mfcc
     */
    public static ByteBuffer loadMFCCBuffer(float[][] array) {

        /* Float Byte size * RGBA * width * height */
        ByteBuffer dataBuf = ByteBuffer.allocateDirect(4 * array.length * array[0].length);
        for (float[] floats : array) {
            for (float aFloat : floats) {
                dataBuf.put(ByteBuffer.allocate(4).putFloat(aFloat).array(), 0, 4);
            }
        }
        dataBuf.position(0);

        return dataBuf;

    }


}
