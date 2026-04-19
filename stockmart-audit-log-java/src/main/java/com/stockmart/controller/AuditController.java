package com.stockmart.controller;

import com.stockmart.model.*;
import com.stockmart.service.AuditService;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import java.time.Instant;
import java.util.*;
import java.nio.charset.StandardCharsets;

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
        safeUserId = safeUserId.replaceAll("^[._-]+|[._-]+$", "");
        if (safeUserId.length() > 64) {
            safeUserId = safeUserId.substring(0, 64);
        }
        if (safeUserId.isEmpty()) {
            safeUserId = "unknown";
        }
        String filename = "audit_" + safeUserId + ".csv";
        String contentDisposition = ContentDisposition.attachment()
            .filename(filename, StandardCharsets.UTF_8)
            .build()
            .toString();
        return ResponseEntity.ok()
            .header(HttpHeaders.CONTENT_TYPE, "text/csv")
            .header(HttpHeaders.CONTENT_DISPOSITION, contentDisposition)
            .body(svc.exportCsv(userId));
    }
}
