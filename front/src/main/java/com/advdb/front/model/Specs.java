package com.advdb.front.model;

public record Specs(
    Long id,
    boolean modText,
    boolean modImage,
    boolean modVideo,
    boolean modAudio,
    boolean modCode,
    boolean modDesign,
    boolean modInfra,
    boolean modProductivity,
    boolean modSafety,
    boolean modMultimodal,
    Integer modalityCount
) {}