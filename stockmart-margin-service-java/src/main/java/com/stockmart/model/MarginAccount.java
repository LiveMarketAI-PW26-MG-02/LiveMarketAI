package com.stockmart.model;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "margin_accounts")
public class MarginAccount {
    @Id
    private String id = UUID.randomUUID().toString();

    @Column(nullable = false, unique = true)
    private String userId;

    private BigDecimal equity        = BigDecimal.ZERO;
    private BigDecimal marginDebt    = BigDecimal.ZERO;
    private BigDecimal maintenanceMarginReq = new BigDecimal("0.25");  // 25% Reg-T
    private BigDecimal initialMarginReq     = new BigDecimal("0.50");  // 50%

    @Enumerated(EnumType.STRING)
    private MarginStatus status = MarginStatus.GOOD;

    private Instant createdAt = Instant.now();
    private Instant updatedAt = Instant.now();

    public BigDecimal buyingPower() {
        // (equity / initial_margin) * 2  for 2:1 leverage
        if (initialMarginReq.compareTo(BigDecimal.ZERO) == 0) return equity;
        return equity.divide(initialMarginReq, 2, RoundingMode.HALF_UP);
    }

    public BigDecimal portfolioValue() {
        return equity.add(marginDebt);
    }

    public BigDecimal marginRatio() {
        BigDecimal pv = portfolioValue();
        if (pv.compareTo(BigDecimal.ZERO) == 0) return BigDecimal.ONE;
        return equity.divide(pv, 4, RoundingMode.HALF_UP);
    }

    public boolean isMarginCall() {
        return marginRatio().compareTo(maintenanceMarginReq) < 0;
    }

    // Getters & Setters
    public String getId() { return id; }
    public String getUserId() { return userId; }
    public void setUserId(String u) { this.userId = u; }
    public BigDecimal getEquity() { return equity; }
    public void setEquity(BigDecimal e) { this.equity = e; }
    public BigDecimal getMarginDebt() { return marginDebt; }
    public void setMarginDebt(BigDecimal d) { this.marginDebt = d; }
    public MarginStatus getStatus() { return status; }
    public void setStatus(MarginStatus s) { this.status = s; }
    public BigDecimal getBuyingPower() { return buyingPower(); }
    public boolean getIsMarginCall() { return isMarginCall(); }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant t) { this.updatedAt = t; }
}
