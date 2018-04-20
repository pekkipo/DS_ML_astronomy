'''
Some reminders of SQL

SELECT kepler_id, COUNT(koi_name)
FROM Planet
GROUP BY kepler_id
HAVING COUNT(koi_name) > 1
ORDER BY COUNT(koi_name) DESC;

gives
+-----------+-------+
| kepler_id | count |
+-----------+-------+
|   4139816 |     4 |
|   8395660 |     4 |
|  10910878 |     3 |
|  10872983 |     3 |
|   5358241 |     3 |

Simple join
SELECT s.radius AS sun_radius,
  p.radius AS planet_radius
FROM Star AS s, Planet AS p
WHERE s.kepler_id = p.kepler_id AND
  s.radius > p.radius
ORDER BY S.radius DESC;

JOIN ... USING:
Specifying a field of attribute to test for equality
JOIN ... ON:
Specifying a condition

Query which counts the number of planets in each solar system where the corresponding stars are larger than our sun (i.e. their radius is larger than 1):

SELECT Star.radius, COUNT(Planet.koi_name)
FROM Star
JOIN Planet USING (kepler_id)
WHERE Star.radius >= 1
GROUP BY Star.kepler_id
HAVING COUNT(Planet.koi_name) > 1
ORDER BY Star.radius DESC;

other than that:
<table1> LEFT OUTER JOIN <table2>
Here all rows from <table1> are kept and missing matches from <table2> are replaced with NULL values.

<table1> RIGHT OUTER JOIN <table2>
All rows from <table2> are kept and missing matches from <table1> are replaced with NULL values.

<table1> FULL OUTER JOIN <table2>
All rows from both tables are kept.

'''