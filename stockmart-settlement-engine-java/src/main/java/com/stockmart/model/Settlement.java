package com.stockmart.model;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "settlements")
public class Settlement {
    @Id
    private String id = UUID.randomUUID().toString();

    private String tradeId;
    private String buyerAccountId;
    private String sellerAccountId;
    private String symbol;
    private BigDecimal quantity;
    private BigDecimal settlementAmount;

    @Enumerated(EnumType.STRING)
    private SettlementStatus status = SettlementStatus.PENDING;

    private LocalDate tradeDate;
    private LocalDate settlementDate;   // T+2
    private int retryCount = 0;
    private String failureReason;
    private Instant createdAt = Instant.now();
    private Instant updatedAt = Instant.now();

    // Getters & Setters
    public String getId() { return id; }
    public String getTradeId() { return tradeId; }
    public void setTradeId(String t) { this.tradeId = t; }
    public String getBuyerAccountId() { return buyerAccountId; }
    public void setBuyerAccountId(String b) { this.buyerAccountId = b; }
    public String getSellerAccountId() { return sellerAccountId; }
    public void setSellerAccountId(String s) { this.sellerAccountId = s; }
    public String getSymbol() { return symbol; }
    public void setSymbol(String s) { this.symbol = s; }
    public BigDecimal getQuantity() { return quantity; }
    public void setQuantity(BigDecimal q) { this.quantity = q; }
    public BigDecimal getSettlementAmount() { return settlementAmount; }
    public void setSettlementAmount(BigDecimal a) { this.settlementAmount = a; }
    public SettlementStatus getStatus() { return status; }
    public void setStatus(SettlementStatus s) { this.status = s; }
    public LocalDate getTradeDate() { return tradeDate; }
    public void setTradeDate(LocalDate d) { this.tradeDate = d; }
    public LocalDate getSettlementDate() { return settlementDate; }
    public void setSettlementDate(LocalDate d) { this.settlementDate = d; }
    public int getRetryCount() { return retryCount; }
    public void setRetryCount(int r) { this.retryCount = r; }
    public String getFailureReason() { return failureReason; }
    public void setFailureReason(String f) { this.failureReason = f; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant t) { this.updatedAt = t; }
}
