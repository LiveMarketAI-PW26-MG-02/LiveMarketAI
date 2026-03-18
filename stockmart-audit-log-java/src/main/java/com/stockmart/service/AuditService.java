package com.stockmart.service;

import com.stockmart.model.*;
import com.stockmart.repository.AuditRepository;
import org.springframework.stereotype.Service;
import java.time.Instant;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class AuditService {
    private final AuditRepository repo;

    public AuditService(AuditRepository repo) { this.repo = repo; }

    public AuditEvent log(String userId, EventType type, String entityType,
                           String entityId, String action, String symbol, String payload) {
        AuditEvent e = new AuditEvent();
        e.setUserId(userId);
        e.setEventType(type);
        e.setEntityType(entityType);
        e.setEntityId(entityId);
        e.setAction(action);
        e.setSymbol(symbol);
        e.setPayload(payload);
        return repo.save(e);
    }

    public List<AuditEvent> getByUser(String userId) {
        return repo.findByUserIdOrderByOccurredAtDesc(userId);
    }

    public List<AuditEvent> getByDateRange(Instant from, Instant to) {
        return repo.findByDateRange(from, to);
    }

    public String exportCsv(String userId) {
        List<AuditEvent> events = getByUser(userId);
        StringBuilder sb = new StringBuilder("id,eventType,action,entityType,entityId,symbol,occurredAt\n");
        events.forEach(e -> sb.append(String.join(",",
            e.getId(), e.getEventType().name(), e.getAction(),
            e.getEntityType(), e.getEntityId(), e.getSymbol() != null ? e.getSymbol() : "",
            e.getOccurredAt().toString()
        )).append("\n"));
        return sb.toString();
    }
}
