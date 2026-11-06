-- Q1: Customers whose completed orders
-- above INR 500 total more than INR 8000

SELECT
    customer_id,
    SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;
-- Q2: Revenue by city and product category

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY
    c.city,
    p.category
ORDER BY
    c.city,
    total_revenue DESC;

-- Q3: Products priced above average catalogue price

SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;
-- Q4: Customers with no completed orders

SELECT
    c.customer_id,
    c.city
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;
-- Q5: Count customers spending above the average

WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
)
SELECT
    COUNT(*) AS customers_above_average,
    ROUND(
        (SELECT AVG(total_spend) FROM customer_spend),
        2
    ) AS average_spend
FROM customer_spend
WHERE total_spend > (
    SELECT AVG(total_spend)
    FROM customer_spend
);
-- Q6: Rank customers by completed spend within each city

WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
ranked_customers AS (
    SELECT
        c.city,
        cs.customer_id,
        ROUND(cs.total_spend, 2) AS total_spend,

        RANK() OVER (
            PARTITION BY c.city
            ORDER BY cs.total_spend DESC
        ) AS spend_rank,

        DENSE_RANK() OVER (
            PARTITION BY c.city
            ORDER BY cs.total_spend DESC
        ) AS dense_spend_rank,

        ROW_NUMBER() OVER (
            PARTITION BY c.city
            ORDER BY cs.total_spend DESC, cs.customer_id
        ) AS row_num

    FROM customer_spend AS cs
    JOIN customers AS c
        ON cs.customer_id = c.customer_id
)
SELECT *
FROM ranked_customers
WHERE spend_rank <= 3 OR row_num <= 3
ORDER BY city, row_num;
-- Q7: Daily completed revenue and cumulative revenue

WITH daily_revenue AS (
    SELECT
        substr(order_ts, 1, 10) AS order_date,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT
    order_date,
    ROUND(daily_revenue, 2) AS daily_revenue,
    ROUND(
        SUM(daily_revenue) OVER (
            ORDER BY order_date
        ),
        2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_date;