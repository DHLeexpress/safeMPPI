# WALLS-4 collision-location audit

Each colliding episode is assigned to the obstacle with the deepest penetration. The four added boundary plugs are labeled separately from interior obstacles.

| Method | Collision episodes / 700 | Interior locations | Wall-plug locations | Max depth (m) |
|---|---:|---|---|---:|
| Full pipeline it140 M25 | 31 | (2.00,2.00): 1, (1.00,1.00): 23, (3.00,3.00): 2, (4.00,4.00): 5 | -- | 0.0361 |
| No curriculum it140 M25 | 37 | (3.00,3.00): 12, (4.00,4.00): 17, (1.00,1.00): 3, (2.00,2.00): 3 | (5.20,4.62): 1, (4.62,5.20): 1 | 0.0205 |
