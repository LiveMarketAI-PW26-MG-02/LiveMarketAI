package com.stockmart.service;

import com.stockmart.model.*;
import com.stockmart.repository.MarginAccountRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.math.BigDecimal;
import java.time.Instant;

@Service
public class MarginService {
    private final MarginAccountRepository repo;

    public MarginService(MarginAccountRepository repo) { this.repo = repo; }

    @Transactional
    public MarginAccount openAccount(String userId, BigDecimal initialEquity) {
        MarginAccount acc = new MarginAccount();
        acc.setUserId(userId);
        acc.setEquity(initialEquity);
        return repo.save(acc);
    }

    @Transactional
    public MarginAccount borrowMargin(String userId, BigDecimal amount) {
        MarginAccount acc = requireAccount(userId);
        acc.setMarginDebt(acc.getMarginDebt().add(amount));
        updateStatus(acc);
        acc.setUpdatedAt(Instant.now());
        return repo.save(acc);
    }

    @Transactional
    public MarginAccount repayMargin(String userId, BigDecimal amount) {
        MarginAccount acc = requireAccount(userId);
        BigDecimal repay = amount.min(acc.getMarginDebt());
        acc.setMarginDebt(acc.getMarginDebt().subtract(repay));
        acc.setEquity(acc.getEquity().subtract(repay));
        updateStatus(acc);
        acc.setUpdatedAt(Instant.now());
        return repo.save(acc);
    }

    @Transactional
    public MarginAccount updateEquity(String userId, BigDecimal newEquity) {
        MarginAccount acc = requireAccount(userId);
        acc.setEquity(newEquity);
        updateStatus(acc);
        acc.setUpdatedAt(Instant.now());
        return repo.save(acc);
    }

    public MarginAccount getAccount(String userId) { return requireAccount(userId); }

    private void updateStatus(MarginAccount acc) {
        if (acc.isMarginCall()) {
            acc.setStatus(MarginStatus.MARGIN_CALL);
        } else if (acc.marginRatio().compareTo(new BigDecimal("0.30")) < 0) {
            acc.setStatus(MarginStatus.WARNING);
        } else {
            acc.setStatus(MarginStatus.GOOD);
        }
    }

    private MarginAccount requireAccount(String userId) {
        return repo.findByUserId(userId)
            .orElseThrow(() -> new IllegalArgumentException("Account not found: " + userId));
    }
}
