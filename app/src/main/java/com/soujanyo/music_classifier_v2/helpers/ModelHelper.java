package com.soujanyo.music_classifier_v2.helpers;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.widget.ImageView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.soujanyo.music_classifier_v2.ml.BollywoodModel;
import com.soujanyo.music_classifier_v2.ml.FMA;
import com.soujanyo.music_classifier_v2.ml.RegionalDataModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelHelper {

    public static final String MODEL_REGIONAL = "regional";
    public static final String MODEL_KAGGLE = "kaggle";
    public static final String MODEL_FMA = "fma";

    private Object model;
    private String modelType;

    private final Context context;

    private static final int NUM_SAMPLES_REGIONAL_MFCC = 1;
    private static final int LEN_MFCC = 130;
    private static final int NUM_MFCC = 14;
    private static final int NUM_COLOR_CHANNELS = 1;

    private final Python python;


    /**
     * @param context input context of the activity calling the model helper class
     */
    public ModelHelper(Context context, String model_type) {
        this.context = context;

        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(context));
        }

        this.python = Python.getInstance();

        if (model_type.equals(MODEL_REGIONAL)) {
            initializeRegionalModel();
        }

        if(model_type.equals(MODEL_KAGGLE)) {
            initializeKaggleModel();
        }

        if(model_type.equals(MODEL_FMA)) {
            initializeFMAModel();
        }
    }


    /**
     * method to initialize regional dataset model
     */
    private void initializeRegionalModel() {
        try {
            this.model = RegionalDataModel.newInstance(this.context);
            this.modelType = MODEL_REGIONAL;
        } catch (IOException e) {
            Util.logInfo("ModelHelper", e.getMessage());

            e.printStackTrace();
        }
    }

    /**
     * method to initialize kaggle dataset model
     */
    private void initializeKaggleModel() {
        try {
            this.model = BollywoodModel.newInstance(this.context);
            this.modelType = MODEL_KAGGLE;
        } catch (IOException e) {
            Util.logInfo("ModelHelper", e.getMessage());

            e.printStackTrace();
        }
    }

    /**
     * method to initialize fma dataset model
     */
    private void initializeFMAModel() {
        try {
            this.model = FMA.newInstance(this.context);
            this.modelType = MODEL_FMA;
        } catch (IOException e) {
            Util.logInfo("ModelHelper", e.getMessage());

            e.printStackTrace();
        }
    }

    public Object all_mfcc(String filePath) {
        PyObject librosaHelper = this.python.getModule("librosa_helper");

        PyObject output = librosaHelper.callAttr("get_all_mfcc", filePath);

        assert output != null;

        return output.toJava(float[][][].class);
    }

    /**
     * @param typeOfFeature type of feature i.e., mfcc, stft, mel spectrogram
     * @param filePath      path of the file selected
     * @return return object mfcc in this case
     */
    public Pair<Object, byte[]> preprocess(int typeOfFeature, String filePath) {

        PyObject librosaHelper = this.python.getModule("librosa_helper");

        PyObject output = null;

        int MFCC = 2;
        if (typeOfFeature == MFCC) {
            output = librosaHelper.callAttr("generate_mfcc", filePath);
        }

        assert output != null;
        List<PyObject> outputList = output.asList();


        return new Pair<>(outputList.get(1).toJava(float[][].class), outputList.get(0).toJava(byte[].class));
    }

    /**
     * @param filePath path of the file selected
     * @return return byte array of image of mfcc
     */
    public byte[] getMFCCImage(String filePath) {

        PyObject librosaHelper = this.python.getModule("librosa_helper");

        PyObject output = librosaHelper.callAttr("get_mfcc_image", filePath);

        return output.toJava(byte[].class);

    }

    public double getPyin(String filePath) {

        PyObject librosaHelper = this.python.getModule("librosa_helper");

        PyObject output = librosaHelper.callAttr("get_pyin", filePath);

        List<PyObject> list = output.asList();

        float result = list.get(0).toJava(float.class);
        float[] f0 = list.get(1).toJava(float[].class);

        Util.logInfo("ModelHelper", Arrays.toString(f0));


        return result;

    }


    /**
     * @param imageView input imageview where image is to be displayed
     * @param frameData input frame data of the image
     */
    public static void displayImage(ImageView imageView, byte[] frameData) {

        Bitmap bitmap = BitmapFactory.decodeByteArray(frameData, 0, frameData.length);
        imageView.setImageBitmap(bitmap);

    }


    /**
     * @param mfcc input mfcc feature for image
     * @return result of the model chosen as a String
     */
    public String run(float[][] mfcc) {

        // initialize input feature
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(
                new int[]{
                        NUM_SAMPLES_REGIONAL_MFCC,
                        LEN_MFCC,
                        NUM_MFCC,
                        NUM_COLOR_CHANNELS
                }, DataType.FLOAT32
        );

        // load input buffer into input feature
        inputFeature0.loadBuffer(Util.loadMFCCBuffer(mfcc));

        TensorBuffer tensorBuffer = null;

        RegionalDataModel regionalDataModel;
        BollywoodModel bollywoodModel;
        FMA fma;

        // get output from the model
        if (this.modelType.equals(MODEL_REGIONAL)) {
             regionalDataModel = (RegionalDataModel) this.model;
            RegionalDataModel.Outputs outputs = regionalDataModel.process(inputFeature0);

            // get buffer from output
            tensorBuffer = outputs.getOutputFeature0AsTensorBuffer();
        }

        if (this.modelType.equals(MODEL_KAGGLE)) {
            bollywoodModel = (BollywoodModel) this.model;
            BollywoodModel.Outputs outputs = bollywoodModel.process(inputFeature0);

            // get buffer from output
            tensorBuffer = outputs.getOutputFeature0AsTensorBuffer();
        }

        if (this.modelType.equals(MODEL_FMA)) {
            fma = (FMA) this.model;
            FMA.Outputs outputs = fma.process(inputFeature0);

            // get buffer from output
            tensorBuffer = outputs.getOutputFeature0AsTensorBuffer();
        }


        // get float result
        assert tensorBuffer != null;
        float[] results = tensorBuffer.getFloatArray();

        // get max probable result
        int maxIndex = Util.getIndexOfMax(results);

        String genre = null;

        // get genre from the max result
        if (this.modelType.equals(MODEL_REGIONAL)) {
            genre = getRegionalSongGenreAtIndex(maxIndex);
            Util.logInfo("ModelHelper", "regional: " + genre);
        }

        if (this.modelType.equals(MODEL_KAGGLE)) {
            genre = getBollywoodSongGenreAtIndex(maxIndex);
            Util.logInfo("ModelHelper", "kaggle: " + genre);
        }

        if (this.modelType.equals(MODEL_FMA)) {
            genre = getFMASongGenreAtIndex(maxIndex);
            Util.logInfo("ModelHelper", "fma: " + genre);
        }


        return genre;



    }

    /**
     * @param index input position of max element of regional model output
     * @return name of genre specific to the model
     */
    private String getFMASongGenreAtIndex(int index) {
        String[] fmaSongsList = new String[]{
                "Avant-Garde",
                "International",
                "Blues",
                "Jazz",
                "Novelty",
                "Classical",
                "Comedy",
                "Old-Time / Historic",
                "Country",
                "Pop",
                "Disco",
                "Rock",
                "Easy Listening",
                "Soul-RnB",
                "Electronic",
                "Sound Effects",
                "Folk",
                "Soundtrack",
                "Funk",
                "Spoken",
                "Hip-Hop",
                "Audio Collage",
                "Punk",
                "Post-Rock",
                "Lo-Fi",
                "Field Recordings",
                "Metal",
                "Noise",
                "Psych-Folk",
                "Krautrock",
                "Jazz: Vocal",
                "Experimental",
                "Electroacoustic",
                "Ambient Electronic",
                "Radio Art",
                "Loud-Rock",
                "Latin America",
                "Drone",
                "Free-Folk",
                "Noise-Rock",
                "Psych-Rock",
                "Bluegrass",
                "Electro-Punk",
                "Radio",
                "Indie-Rock",
                "Industrial",
                "No Wave",
                "Free-Jazz",
                "Experimental Pop",
                "French",
                "Reggae - Dub",
                "Afrobeat",
                "Nerdcore",
                "Garage",
                "Indian",
                "New Wave",
                "Post-Punk",
                "Sludge",
                "African",
                "Freak-Folk",
                "Jazz: Out",
                "Progressive",
                "Alternative Hip-Hop",
                "Death-Metal",
                "Middle East",
                "Singer-Songwriter",
                "Ambient",
                "Hardcore",
                "Power-Pop",
                "Space-Rock",
                "Polka",
                "Balkan",
                "Unclassifiable",
                "Europe",
                "Americana",
                "Spoken Weird",
                "Interview",
                "Black-Metal",
                "Rockabilly",
                "Easy Listening: Vocal",
                "Brazilian",
                "Asia-Far East",
                "N. Indian Traditional",
                "South Indian Traditional",
                "Bollywood",
                "Pacific",
                "Celtic",
                "Be-Bop",
                "Big Band/Swing",
                "British Folk",
                "Techno",
                "House",
                "Glitch",
                "Minimal Electronic",
                "Breakcore - Hard",
                "Sound Poetry",
                "20th Century Classical",
                "Poetry",
                "Talk Radio",
                "North African",
                "Sound Collage",
                "Flamenco",
                "IDM",
                "Chiptune",
                "Musique Concrete",
                "Improv",
                "New Age",
                "Trip-Hop",
                "Dance",
                "Chip Music",
                "Lounge",
                "Goth",
                "Composed Music",
                "Drum & Bass",
                "Shoegaze",
                "Kid-Friendly",
                "Thrash",
                "Synth Pop",
                "Banter",
                "Deep Funk",
                "Spoken Word",
                "Chill-out",
                "Bigbeat",
                "Surf",
                "Radio Theater",
                "Grindcore",
                "Rock Opera",
                "Opera",
                "Chamber Music",
                "Choral Music",
                "Symphony",
                "Minimalism",
                "Musical Theater",
                "Dubstep",
                "Skweee",
                "Western Swing",
                "Downtempo",
                "Cumbia",
                "Latin",
                "Sound Art",
                "Romany (Gypsy)",
                "Compilation",
                "Rap",
                "Breakbeat",
                "Gospel",
                "Abstract Hip-Hop",
                "Reggae - Dancehall",
                "Spanish",
                "Country & Western",
                "Contemporary Classical",
                "Wonky",
                "Jungle",
                "Klezmer",
                "Holiday",
                "Salsa",
                "Nu-Jazz",
                "Hip-Hop Beats",
                "Modern Jazz",
                "Turkish",
                "Tango",
                "Fado",
                "Christmas",
                "Instrumental"
        };

        return fmaSongsList[index];
    }

    /**
     * @param index input position of max element of regional model output
     * @return name of genre specific to the model
     */
    private String getBollywoodSongGenreAtIndex(int index) {
        String[] bollywoodSongsList = new String[]{
                "Semi-Classical",
                "Ghazal",
                "Carnatic",
                "Bollypop",
                "Sufi"
        };


        return bollywoodSongsList[index];
    }

    /**
     * @param index input position of max element of regional model output
     * @return name of genre specific to the model
     */
    private String getRegionalSongGenreAtIndex(int index) {

        String[] regionalSongsList = new String[]{
                "Nepali",
                "Kannada",
                "Tamil",
                "Punjabi",
                "Malayalam",
                "Telugu",
                "Assamese",
                "Kashmiri",
                "Bengali",
                "Manipuri",
                "Konkani",
                "Gujrati",
                "Marathi",
                "Khasi and Jaintia",
                "Hindi",
                "Nagamese",
                "Oriya"
        };


        return regionalSongsList[index];


    }

    public float getBPM(String file_path) {
        PyObject librosaHelper = this.python.getModule("librosa_helper");

        PyObject tempo = librosaHelper.callAttr("get_tempo", file_path);

        return tempo.toJava(float.class);

    }


    public void closeModel() {

        if(this.modelType.equals(MODEL_REGIONAL)) {
            RegionalDataModel model = (RegionalDataModel) this.model;
            model.close();
        }

        if(this.modelType.equals(MODEL_KAGGLE)) {
            BollywoodModel model = (BollywoodModel) this.model;
            model.close();
        }

        if(this.modelType.equals(MODEL_FMA)) {
            FMA model = (FMA) this.model;
            model.close();
        }


    }


}
