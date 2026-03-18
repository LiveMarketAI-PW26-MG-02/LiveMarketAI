package com.stockmart.repository;

import com.stockmart.model.AuditEvent;
import com.stockmart.model.EventType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import java.time.Instant;
import java.util.List;

public interface AuditRepository extends JpaRepository<AuditEvent, String> {
    List<AuditEvent> findByUserIdOrderByOccurredAtDesc(String userId);
    List<AuditEvent> findBySymbolAndEventType(String symbol, EventType type);

    @Query("SELECT e FROM AuditEvent e WHERE e.occurredAt BETWEEN :from AND :to ORDER BY e.occurredAt DESC")
    List<AuditEvent> findByDateRange(Instant from, Instant to);
}
