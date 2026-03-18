package com.stockmart.repository;

import com.stockmart.model.MarginAccount;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface MarginAccountRepository extends JpaRepository<MarginAccount, String> {
    Optional<MarginAccount> findByUserId(String userId);
}
