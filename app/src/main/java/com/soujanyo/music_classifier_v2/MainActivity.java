package com.soujanyo.music_classifier_v2;

import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Pair;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.anggrayudi.storage.SimpleStorageHelper;
import com.anggrayudi.storage.file.DocumentFileUtils;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.soujanyo.music_classifier_v2.helpers.ModelHelper;
import com.soujanyo.music_classifier_v2.helpers.Util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements View.OnClickListener, AdapterView.OnItemSelectedListener {

    // constants for selecting model
    private static final int REGIONAL = 0;
    private static final int KAGGLE = 1;
    private static final int FMA = 2;

    // model selected at run time
    private int modelSelected;

    // xml
    private ImageView folderPickerButton, showResultImageView;
    private Spinner selectModelSpinner;
    private TextView showFileName;

    // simple storage
    private SimpleStorageHelper simpleStorageHelper;

    // model helper
    private ModelHelper modelHelper;

    // file selection
    private String songSelected;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // get ids
        getIds();
        setClickListeners();


        // simple storage
        simpleStorageHelper = new SimpleStorageHelper(this);
        setupStorageHelper();

        setupSpinner();

        // model
        modelHelper = new ModelHelper(this, ModelHelper.MODEL_REGIONAL);

    }

    // get all ids
    private void getIds() {
        this.folderPickerButton = findViewById(R.id.filePickerButton);
        this.showResultImageView = findViewById(R.id.showResultImageView);
        this.selectModelSpinner = findViewById(R.id.selectModelSpinner);
        this.showFileName = findViewById(R.id.showFileName);
    }

    // click listener
    private void setClickListeners() {
        this.folderPickerButton.setOnClickListener(this);
        this.showResultImageView.setOnClickListener(this);
    }

    /**
     * method to setup spinner private to main activity
     */
    private void setupSpinner() {

        String[] models = new String[]{"Select Model", "Regional Music", "Bollywood Music", "Western Music"};
        ArrayAdapter<String> spinnerAdapter = new ArrayAdapter<>(MainActivity.this, android.R.layout.simple_spinner_dropdown_item, models);

        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        this.selectModelSpinner.setAdapter(spinnerAdapter);
        this.selectModelSpinner.setOnItemSelectedListener(this);

    }

    /**
     * setup storage helper for the app
     */
    private void setupStorageHelper() {

        simpleStorageHelper.setOnFileSelected((requestCode, files) -> {

            if (!Python.isStarted()) {
                Python.start(new AndroidPlatform(this));
            }

            this.songSelected = DocumentFileUtils.getAbsolutePath(files.get(0), this);

            showFileName();

            return null;
        });

    }

    /**
     * show file name for the particular file selected
     */
    private void showFileName() {

        this.showFileName.setText(this.songSelected);
        this.showFileName.setVisibility(View.VISIBLE);

    }


    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {

        switch (i) {
            case 1:
                this.modelSelected = REGIONAL;
                break;
            case 2:
                this.modelSelected = KAGGLE;
                break;
            case 3:
                this.modelSelected = FMA;
                break;
            default:
                this.modelSelected = -1;
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {
        Toast.makeText(this, "Please select something!", Toast.LENGTH_SHORT).show();
    }

    // for simple storage
    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        simpleStorageHelper.onSaveInstanceState(outState);
        super.onSaveInstanceState(outState);
    }

    @Override
    protected void onRestoreInstanceState(@NonNull Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        simpleStorageHelper.onRestoreInstanceState(savedInstanceState);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        simpleStorageHelper.getStorage().onActivityResult(requestCode, resultCode, data);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        simpleStorageHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);

        SharedPreferences preferences = MainActivity.this.getPreferences(Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = preferences.edit();
        editor.putBoolean("storageAccessGranted", true);
        editor.apply();

    }

    private void showResult() {


        if(modelSelected == REGIONAL) {
            modelHelper = new ModelHelper(this, ModelHelper.MODEL_REGIONAL);
        }

        if(modelSelected == KAGGLE) {
            modelHelper = new ModelHelper(this, ModelHelper.MODEL_KAGGLE);
        }

        if (modelSelected == FMA) {
            modelHelper = new ModelHelper(this, ModelHelper.MODEL_FMA);
        }


        ProgressDialog progressDialog =  new ProgressDialog(this);
        progressDialog.setTitle("Running Model");
        progressDialog.setMax(100);
        progressDialog.setCancelable(false);
        progressDialog.setMessage("Initializing....");
        progressDialog.show();


        new Thread(() -> {

            progressDialog.setMessage("Processing input...");

            byte[] frameData = modelHelper.getMFCCImage(songSelected);

            double fundamentalFrequency = modelHelper.getPyin(songSelected);

            Util.logInfo("MainActivity", Double.toString(fundamentalFrequency));

            int tempo = Math.round(modelHelper.getBPM(songSelected));

            float[][][] mfcc_list = (float[][][]) modelHelper.all_mfcc(songSelected);


            ArrayList<String> genres = new ArrayList<>();
            for (float[][] mfcc : mfcc_list) {
                String genre = modelHelper.run(mfcc);
                Util.logInfo("Main", genre);
                genres.add(genre);
            }

            modelHelper.closeModel();

            String genre = getMaxFrequency(genres);


            Intent intent = new Intent(MainActivity.this, ResultActivity.class);
            intent.putExtra("genre", genre);
            intent.putExtra("image", frameData);
            intent.putExtra("tempo", tempo);


            genres.clear();

            Util.logInfo("MainActivity", genres.toString());

            startActivity(intent);

            progressDialog.dismiss();


        }).start();
    }


    /**
     * @param list list of genres outputted by model
     * @return item with max frequency from the list
     */
    private String getMaxFrequency(ArrayList<String> list) {

        Map<String, Integer> map = new HashMap<>();

        for (String item: list) {
            Integer value = map.get(item);
            map.put(item, value == null ? 1 : value + 1);
        }

        Map.Entry<String, Integer> max = null;

        for (Map.Entry<String, Integer> e : map.entrySet()) {
            if(max == null || e.getValue() > max.getValue()) {
                max = e;
            }
        }

        assert max != null;
        return max.getKey();

    }


    @Override
    public void onClick(View view) {

        int id = view.getId();

        if (id == R.id.filePickerButton) {
            // open file picker
            if (!getPreferences(Context.MODE_PRIVATE).getBoolean("storageAccessGranted", false)) {
                simpleStorageHelper.requestStorageAccess();
            }
            simpleStorageHelper.openFilePicker("audio/x-wav");
        }

        if (id == R.id.showResultImageView) {

            if(this.songSelected == null) {
                Toast.makeText(this, "Please select a file!", Toast.LENGTH_SHORT).show();
            } else if (this.modelSelected == -1) {
                Toast.makeText(this, "Please select a proper model to run on!", Toast.LENGTH_SHORT).show();
            } else {
                showResult();

            }

        }

    }

}