-- creates the MySQL server user
-- MySQL server
SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
JOIN tv_show_genres
LEFT JOIN tv_genres
ON tv_genres.id WHERE tv_genres.id = tv_show_genres.genre_id AND tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
