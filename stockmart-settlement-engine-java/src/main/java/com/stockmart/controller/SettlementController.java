package com.stockmart.controller;

import com.stockmart.model.Settlement;
import com.stockmart.service.SettlementService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/settlements")
public class SettlementController {
    private final SettlementService svc;
    public SettlementController(SettlementService svc) { this.svc = svc; }

    @PostMapping
    public ResponseEntity<Settlement> create(@RequestBody Settlement s) {
        return ResponseEntity.status(201).body(svc.createSettlement(s));
    }

    @GetMapping
    public List<Settlement> getAll() { return svc.getAll(); }

    @GetMapping("/pending")
    public List<Settlement> getPending() { return svc.getPending(); }

    @PostMapping("/{id}/settle")
    public ResponseEntity<Void> manualSettle(@PathVariable String id) {
        svc.getAll().stream().filter(s -> s.getId().equals(id))
           .findFirst().ifPresent(svc::settle);
        return ResponseEntity.ok().build();
    }
}
