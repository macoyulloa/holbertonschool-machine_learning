-- average temparute
-- MySQL server
SELECT city, AVG(value) AS avg_tem FROM temperatures GROUP BY city ORDER BY 2 DESC;
