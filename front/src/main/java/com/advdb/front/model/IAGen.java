package com.advdb.front.model;

import java.time.LocalDate;

public record IAGen(
    Long id,
    String name,
    String website,
    LocalDate releaseYear,
    Long fkSpecs,
    Long fkCompany,
    Long fkIatype,
    Long fkCfg
) {}