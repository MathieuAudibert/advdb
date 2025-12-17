package com.advdb.front.model;

import java.time.LocalDateTime;

public record LogsIA(
    Long id,
    String type,
    LocalDateTime dateCreation,
    String state
) {}