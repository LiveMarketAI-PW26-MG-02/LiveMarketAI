package com.stockmart.repository;

import com.stockmart.model.Settlement;
import com.stockmart.model.SettlementStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import java.time.LocalDate;
import java.util.List;

public interface SettlementRepository extends JpaRepository<Settlement, String> {
    List<Settlement> findBySettlementDateAndStatus(LocalDate date, SettlementStatus status);
    List<Settlement> findByStatus(SettlementStatus status);
}
