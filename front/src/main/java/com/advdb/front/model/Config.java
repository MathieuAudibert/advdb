package com.advdb.front.model;

public record Config(
    Long id,
    boolean apiAvailable,
    boolean openSource
) {}