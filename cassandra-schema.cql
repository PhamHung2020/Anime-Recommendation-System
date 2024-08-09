-- Create a keyspace
CREATE KEYSPACE IF NOT EXISTS anime WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : '1' };

-- Create a table
CREATE TABLE IF NOT EXISTS anime.test_animes (
    anime_id int PRIMARY KEY,
    Name text,
    Score float,
);

CREATE TABLE IF NOT EXISTS anime.anime_similarity (
    anime_id int,
    anime_id_2 int,
    cos_sim float,
    PRIMARY KEY (anime_id, anime_id_2)
);