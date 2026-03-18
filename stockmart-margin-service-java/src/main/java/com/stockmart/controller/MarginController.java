package com.stockmart.controller;

import com.stockmart.model.MarginAccount;
import com.stockmart.service.MarginService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.math.BigDecimal;
import java.util.Map;

@RestController
@RequestMapping("/margin")
public class MarginController {
    private final MarginService svc;
    public MarginController(MarginService svc) { this.svc = svc; }

    @PostMapping("/accounts")
    public ResponseEntity<MarginAccount> open(@RequestBody Map<String, String> body) {
        return ResponseEntity.status(201).body(
            svc.openAccount(body.get("userId"), new BigDecimal(body.get("equity")))
        );
    }

    @GetMapping("/accounts/{userId}")
    public MarginAccount getAccount(@PathVariable String userId) {
        return svc.getAccount(userId);
    }

    @PostMapping("/accounts/{userId}/borrow")
    public MarginAccount borrow(@PathVariable String userId, @RequestBody Map<String, String> b) {
        return svc.borrowMargin(userId, new BigDecimal(b.get("amount")));
    }

    @PostMapping("/accounts/{userId}/repay")
    public MarginAccount repay(@PathVariable String userId, @RequestBody Map<String, String> b) {
        return svc.repayMargin(userId, new BigDecimal(b.get("amount")));
    }
}
