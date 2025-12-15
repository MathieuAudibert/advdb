SELECT g.name, g.release_year
FROM IA_Gen g
JOIN Config cfg ON g.fk_config = cfg.id
WHERE cfg.api_available = TRUE AND cfg.open_source = TRUE;
