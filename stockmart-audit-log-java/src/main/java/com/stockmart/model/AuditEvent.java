package com.stockmart.model;

import jakarta.persistence.*;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "audit_events")
public class AuditEvent {
    @Id
    private String id = UUID.randomUUID().toString();

    @Column(nullable = false)
    private String userId;

    @Enumerated(EnumType.STRING)
    private EventType eventType;

    private String entityId;       // orderId, tradeId etc.
    private String entityType;     // Order, Trade, Account
    private String action;         // CREATED, UPDATED, CANCELLED, FILLED
    private String symbol;

    @Column(length = 2000)
    private String payload;        // JSON snapshot

    private String ipAddress;
    private String userAgent;
    private Instant occurredAt = Instant.now();

    // Getters & Setters
    public String getId() { return id; }
    public String getUserId() { return userId; }
    public void setUserId(String u) { this.userId = u; }
    public EventType getEventType() { return eventType; }
    public void setEventType(EventType e) { this.eventType = e; }
    public String getEntityId() { return entityId; }
    public void setEntityId(String e) { this.entityId = e; }
    public String getEntityType() { return entityType; }
    public void setEntityType(String e) { this.entityType = e; }
    public String getAction() { return action; }
    public void setAction(String a) { this.action = a; }
    public String getSymbol() { return symbol; }
    public void setSymbol(String s) { this.symbol = s; }
    public String getPayload() { return payload; }
    public void setPayload(String p) { this.payload = p; }
    public String getIpAddress() { return ipAddress; }
    public void setIpAddress(String ip) { this.ipAddress = ip; }
    public String getUserAgent() { return userAgent; }
    public void setUserAgent(String ua) { this.userAgent = ua; }
    public Instant getOccurredAt() { return occurredAt; }
}
