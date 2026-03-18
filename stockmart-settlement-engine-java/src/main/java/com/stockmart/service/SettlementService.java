package com.stockmart.service;

import com.stockmart.model.*;
import com.stockmart.repository.SettlementRepository;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.*;
import java.util.List;

@Service
public class SettlementService {
    private final SettlementRepository repo;
    private static final int MAX_RETRIES = 3;

    public SettlementService(SettlementRepository repo) { this.repo = repo; }

    @Transactional
    public Settlement createSettlement(Settlement s) {
        s.setTradeDate(LocalDate.now());
        s.setSettlementDate(LocalDate.now().plusDays(2));  // T+2
        return repo.save(s);
    }

    @Transactional
    @Scheduled(cron = "0 0 8 * * MON-FRI")  // 8am weekdays
    public void runDailySettlement() {
        LocalDate today = LocalDate.now();
        List<Settlement> due = repo.findBySettlementDateAndStatus(today, SettlementStatus.PENDING);
        due.forEach(this::settle);
    }

    @Transactional
    public void settle(Settlement s) {
        s.setStatus(SettlementStatus.PROCESSING);
        s.setUpdatedAt(Instant.now());
        try {
            // Simulate DVP check — in prod: debit buyer, credit seller
            if (s.getRetryCount() >= MAX_RETRIES) {
                s.setStatus(SettlementStatus.FAILED);
                s.setFailureReason("Max retries exceeded");
            } else {
                s.setStatus(SettlementStatus.SETTLED);
            }
        } catch (Exception e) {
            s.setRetryCount(s.getRetryCount() + 1);
            s.setStatus(SettlementStatus.PENDING);
            s.setFailureReason(e.getMessage());
        }
        s.setUpdatedAt(Instant.now());
        repo.save(s);
    }

    public List<Settlement> getPending() {
        return repo.findByStatus(SettlementStatus.PENDING);
    }

    public List<Settlement> getAll() { return repo.findAll(); }
}
