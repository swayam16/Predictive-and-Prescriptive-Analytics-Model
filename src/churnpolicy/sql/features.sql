-- Point-in-time feature extraction for one snapshot date (:snapshot, the 1st of a month).
--
-- Rule: every column below is computable on the morning of :snapshot. Nothing dated on or
-- after it is read, and ticket resolution times / CSAT are only used when the ticket was
-- already closed before the snapshot. tests/test_features.py enforces this by deleting all
-- future rows and checking the output does not change.

WITH params AS (
    SELECT date(:snapshot) AS s
),

population AS (                    -- customers active on the snapshot with >= 1 full month of history
    SELECT c.customer_id, c.plan, c.contract, c.channel, c.region, c.signup_date,
           (julianday(p.s) - julianday(c.signup_date)) / 30.4375 AS tenure_months
    FROM customers c
    CROSS JOIN params p
    LEFT JOIN cancellations x ON x.customer_id = c.customer_id AND x.cancelled_at < p.s
    WHERE c.signup_date < date(p.s, '-1 month')
      AND x.customer_id IS NULL
),

usage AS (
    SELECT u.customer_id,
           SUM(CASE WHEN u.month = date(p.s, '-1 month') THEN u.logins END)          AS logins_1m,
           AVG(CASE WHEN u.month BETWEEN date(p.s, '-4 month') AND date(p.s, '-2 month')
                    THEN u.logins END)                                              AS logins_prev3m_avg,
           SUM(CASE WHEN u.month = date(p.s, '-1 month') THEN u.active_days END)     AS active_days_1m,
           AVG(CASE WHEN u.month BETWEEN date(p.s, '-4 month') AND date(p.s, '-2 month')
                    THEN u.active_days END)                                         AS active_days_prev3m_avg,
           SUM(CASE WHEN u.month = date(p.s, '-1 month') THEN u.session_minutes END) AS minutes_1m,
           SUM(CASE WHEN u.month = date(p.s, '-1 month') THEN u.features_used END)   AS features_used_1m,
           SUM(CASE WHEN u.month = date(p.s, '-1 month') THEN u.seats END)           AS seats_now,
           SUM(CASE WHEN u.month = date(p.s, '-4 month') THEN u.seats END)           AS seats_3m_ago,
           MAX(u.last_login_at)                                                     AS last_login_at
    FROM monthly_usage u
    CROSS JOIN params p
    WHERE u.month < p.s
      AND u.month >= date(p.s, '-6 month')
    GROUP BY u.customer_id
),

tickets AS (
    SELECT t.customer_id,
           COUNT(*)                                                       AS tickets_90d,
           SUM(t.priority = 'high')                                       AS high_priority_tickets_90d,
           SUM(julianday(t.opened_at) + t.resolution_hours / 24.0 >= julianday(p.s)) AS open_tickets,
           AVG(CASE WHEN julianday(t.opened_at) + t.resolution_hours / 24.0 < julianday(p.s)
                    THEN t.resolution_hours END)                          AS avg_resolution_hours_90d,
           AVG(CASE WHEN julianday(t.opened_at) + t.resolution_hours / 24.0 < julianday(p.s)
                    THEN t.csat END)                                      AS avg_csat_90d
    FROM support_tickets t
    CROSS JOIN params p
    WHERE t.opened_at >= date(p.s, '-90 day')
      AND t.opened_at < p.s
    GROUP BY t.customer_id
),

billing AS (
    SELECT i.customer_id,
           SUM(i.status = 'failed' AND i.invoice_date >= date(p.s, '-3 month'))       AS failed_payments_90d,
           SUM(CASE WHEN i.invoice_date = date(p.s, '-1 month') THEN i.amount END)     AS monthly_revenue,
           SUM(CASE WHEN i.invoice_date = date(p.s, '-4 month') THEN i.amount END)     AS amount_3m_ago,
           MAX(CASE WHEN i.invoice_date = date(p.s, '-1 month') THEN i.discount_pct END) AS discount_pct
    FROM invoices i
    CROSS JOIN params p
    WHERE i.invoice_date < p.s
      AND i.invoice_date >= date(p.s, '-6 month')
    GROUP BY i.customer_id
)

SELECT p.s AS snapshot,
       pop.customer_id, pop.plan, pop.contract, pop.channel, pop.region, pop.tenure_months,
       u.logins_1m, u.logins_prev3m_avg, u.active_days_1m, u.active_days_prev3m_avg,
       u.minutes_1m, u.features_used_1m, u.seats_now, u.seats_3m_ago,
       julianday(p.s) - julianday(u.last_login_at)                     AS days_since_last_login,
       COALESCE(t.tickets_90d, 0)                                      AS tickets_90d,
       COALESCE(t.high_priority_tickets_90d, 0)                        AS high_priority_tickets_90d,
       COALESCE(t.open_tickets, 0)                                     AS open_tickets,
       t.avg_resolution_hours_90d, t.avg_csat_90d,
       COALESCE(b.failed_payments_90d, 0)                              AS failed_payments_90d,
       b.monthly_revenue, b.amount_3m_ago, COALESCE(b.discount_pct, 0) AS discount_pct
FROM population pop
CROSS JOIN params p
LEFT JOIN usage   u ON u.customer_id = pop.customer_id
LEFT JOIN tickets t ON t.customer_id = pop.customer_id
LEFT JOIN billing b ON b.customer_id = pop.customer_id
ORDER BY pop.customer_id;
