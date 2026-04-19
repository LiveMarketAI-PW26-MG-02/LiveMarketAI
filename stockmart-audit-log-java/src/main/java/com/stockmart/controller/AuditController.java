package com.stockmart.controller;

import com.stockmart.model.*;
import com.stockmart.service.AuditService;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import java.time.Instant;
import java.util.*;

@RestController
@RequestMapping("/audit")
public class AuditController {
    private final AuditService svc;
    public AuditController(AuditService svc) { this.svc = svc; }

    @PostMapping("/events")
    public ResponseEntity<AuditEvent> log(@RequestBody Map<String, String> body) {
        AuditEvent e = svc.log(
            body.get("userId"), EventType.valueOf(body.get("eventType")),
            body.get("entityType"), body.get("entityId"),
            body.get("action"), body.get("symbol"), body.get("payload")
        );
        return ResponseEntity.status(201).body(e);
    }

    @GetMapping("/users/{userId}")
    public List<AuditEvent> byUser(@PathVariable String userId) {
        return svc.getByUser(userId);
    }

    @GetMapping("/users/{userId}/export")
    public ResponseEntity<String> export(@PathVariable String userId) {
        String safeUserId = userId == null ? "" : userId.replaceAll("[^A-Za-z0-9._-]", "_");
        if (safeUserId.isEmpty()) {
            safeUserId = "unknown";
        }
        return ResponseEntity.ok()
            .header(HttpHeaders.CONTENT_TYPE, "text/csv")
            .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=audit_" + safeUserId + ".csv")
            .body(svc.exportCsv(userId));
    }
}
