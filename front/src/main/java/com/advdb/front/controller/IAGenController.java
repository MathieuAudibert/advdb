package com.advdb.front.controller;

import java.time.LocalDateTime;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import com.advdb.front.model.IAGen;

@Controller
@RequestMapping("/ias")
public class IAGenController {
    @Autowired
    private JdbcTemplate jdbcTemplate;

    private LocalDateTime dateCreation;

    @GetMapping
    public String listIAs(Model model) {
        String sql = "SELECT * FROM IA_Gen";
        List<IAGen> ias = jdbcTemplate.query(sql, (rs, rowNum) -> new IAGen(
            rs.getLong("id"),
            rs.getString("name"),
            rs.getString("website"),
            rs.getDate("release_year") != null ? rs.getDate("release_year").toLocalDate() : null,
            rs.getLong("fk_specs"),
            rs.getLong("fk_company"),
            rs.getLong("fk_iatype"),
            rs.getLong("fk_cfg")
        ));
        model.addAttribute("ias", ias);
        return "ia-list"; 
    }

    @PostMapping("/add")
    public String addIA(@RequestParam String name, @RequestParam String website) {
        String sql = "INSERT INTO IA_Gen (name, website) VALUES (?, ?)";
        String logs = "INSERT INTO logs_ia (type, date_creation, state) VALUES (?, ?, ?)";
        jdbcTemplate.update(sql, name, website);
        jdbcTemplate.update(logs, "insertion", dateCreation, "completed");
        return "forward:/ias";
    }

    @GetMapping("/delete/{id}")
    public String deleteIA(@PathVariable Long id) {
        String logs = "INSERT INTO logs_ia (type, date_creation, state) VALUES (?, ?, ?)";
        jdbcTemplate.update("DELETE FROM IA_Gen WHERE id = ?", id);
        jdbcTemplate.update(logs, "deletion", dateCreation, "completed");
        return "forward:/ias";
    }
}