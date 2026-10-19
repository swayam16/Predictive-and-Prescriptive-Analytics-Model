-- Outcome for one snapshot: did the customer cancel within :label_days days after :snapshot?
-- Used only to build training and evaluation targets, never as a feature.
SELECT c.customer_id,
       CASE WHEN x.cancelled_at >= date(:snapshot)
             AND x.cancelled_at <  date(:snapshot, '+' || :label_days || ' day')
            THEN 1 ELSE 0 END AS churned
FROM customers c
LEFT JOIN cancellations x ON x.customer_id = c.customer_id;
