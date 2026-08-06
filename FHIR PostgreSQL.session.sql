SELECT *
FROM messages
WHERE created_at >= DATE '2026-08-05'
  AND created_at <  DATE '2026-08-06'
ORDER BY created_at ASC;