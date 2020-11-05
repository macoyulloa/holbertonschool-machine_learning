-- average temparute
-- MySQL server
SELECT state, MAX(value) AS max_tem FROM temperatures GROUP BY state ORDER BY state;
