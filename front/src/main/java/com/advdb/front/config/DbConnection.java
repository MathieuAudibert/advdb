package com.advdb.front.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import javax.sql.DataSource;
import org.springframework.boot.jdbc.DataSourceBuilder;
import org.springframework.jdbc.core.JdbcTemplate;

@Configuration
public class DbConnection {
    @Value("${oracle.db.url}")
    private String dbUrl;

    @Value("${oracle.db.username}")
    private String username;

    @Value("${oracle.db.password}")
    private String passwd;

    @Value("${oracle.db.driver}")
    private String driver;

    @Bean
    @Primary
    public DataSource dataSource(){
        return DataSourceBuilder.create()
        .url(dbUrl)
        .username(username)
        .password(passwd)
        .driverClassName(driver)
        .build();
    }

    @Bean
    public JdbcTemplate jdbcTemplate(DataSource dataSource){
        return new JdbcTemplate(dataSource);
    }
}
