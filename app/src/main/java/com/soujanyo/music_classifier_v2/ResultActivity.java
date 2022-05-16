package com.soujanyo.music_classifier_v2;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import com.soujanyo.music_classifier_v2.helpers.ModelHelper;
import com.soujanyo.music_classifier_v2.helpers.Util;

public class ResultActivity extends AppCompatActivity {

    // xml
    private ImageView audioFeatureImageView;
    private TextView genreTextView, tempoTextView;

    private String genre;
    private int tempo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);


        byte[] imageData = getIntent().getByteArrayExtra("image");
        genre = getIntent().getStringExtra("genre");
        tempo = getIntent().getIntExtra("tempo", -1);

        // xml
        audioFeatureImageView = findViewById(R.id.audioFeatureImageView);
        genreTextView = findViewById(R.id.genreTextView);
        tempoTextView = findViewById(R.id.tempoTextView);

        ModelHelper.displayImage(audioFeatureImageView, imageData);

        showTable();

    }


    private void showTable() {

        if(genre != null) {
            genreTextView.setText(genre);
        }

        tempoTextView.setText(String.valueOf(tempo));


    }
}