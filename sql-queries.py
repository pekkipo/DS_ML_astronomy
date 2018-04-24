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
JOIN Planet USING (kepler_id)   The inner join is implicit when calling the JOIN
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

SELECT S.kepler_id, P.koi_name
FROM Star S
FULL OUTER JOIN Planet P USING(kepler_id);

Query which returns the kepler_id, t_eff and radius for all stars in the Star table which haven't got a planet as join partner.
LONELY STARS

SELECT s.kepler_id, s.t_eff, s.radius
FROM Star AS s
LEFT OUTER JOIN Planet AS p USING (kepler_id)
WHERE p.koi_name is NULL
ORDER BY t_eff DESC;

NESTED QUERIES AND SUBQUERIES

SELECT * FROM Star
WHERE Star.radius > (
  SELECT AVG(radius) FROM Star
);

Co-related: the subquery is executed for each element of the outer query.
SELECT s.kepler_id
FROM Star s
WHERE EXISTS (
  SELECT * FROM Planet p
  WHERE s.kepler_id = p.kepler_id
    AND p.radius < 1
);

Non-co-related: the subquery is executed only once.
SELECT s.kepler_id
FROM Star s
WHERE s.kepler_id IN (
  SELECT p.kepler_id FROM Planet p
  WHERE p.radius < 1
);


Query which queries both the Star and the Planet table and calculates the following quantities:

the average value of the planets' equilibrium temperature t_eq, rounded to one decimal place;
the minimum effective temperature t_eff of the stars;
the maximum value of t_eff;

SELECT ROUND(AVG(P.t_eq), 1), MIN(S.t_eff), MAX(S.t_eff)
FROM Star S
JOIN Planet P USING(kepler_id)
WHERE S.t_eff > (
  SELECT AVG(t_eff) FROM Star
);

Query which finds the radii of those planets in the Planet table which orbit the five largest stars in the Star table.

SELECT p.koi_name, p.radius, s.radius
FROM Star AS s
JOIN Planet AS p USING (kepler_id)
WHERE s.kepler_id IN (
  SELECT kepler_id
  FROM Star
  ORDER BY radius DESC
  LIMIT 5
);

///

UPDATE Planet
 SET kepler_name = NULL
 WHERE UPPER(status) != 'CONFIRMED';


DELETE FROM Planet
 WHERE radius < 0;


CREATE TABLE Star (
  kepler_id INTEGER PRIMARY KEY,
  t_eff INTEGER CHECK (t_eff > 3000),
  radius FLOAT
);

INSERT INTO Star VALUES
  (10341777, 6302, 0.815);

FOREIGN KEY
CREATE TABLE Star (
  kepler_id INTEGER PRIMARY KEY
);

CREATE TABLE Planet (
  kepler_id INTEGER REFERENCES Star (kepler_id)
);

INSERT INTO Star VALUES (10341777);
INSERT INTO Planet VALUES (10341777);

COPY FROM CSV

CREATE TABLE Star (
  kepler_id INTEGER PRIMARY KEY,
  t_eff INTEGER,
  radius FLOAT
);

COPY Star (kepler_id, t_eff, radius)
  FROM 'stars.csv' CSV;

ALTER TABLE command, which allows us to add, delete and modify the columns in an existing table.

ALTER TABLE Star
ADD COLUMN ra FLOAT,
ADD COLUMN decl FLOAT;

ALTER TABLE Star
 ALTER COLUMN t_eff SET DATA TYPE FLOAT;

ALTER TABLE Star
  ADD CONSTRAINT radius CHECK(radius > 0);

'''