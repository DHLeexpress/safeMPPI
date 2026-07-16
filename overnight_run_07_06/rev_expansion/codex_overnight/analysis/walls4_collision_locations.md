# WALLS-4 collision-location audit

Each colliding episode is assigned to the obstacle with the deepest penetration. The four added boundary plugs are labeled separately from interior obstacles.

| Method | Collision episodes / 700 | Interior locations | Wall-plug locations | Max depth (m) |
|---|---:|---|---|---:|
| Full pipeline it100 | 119 | (2.00,1.00): 1, (4.00,4.00): 22, (1.00,1.00): 70, (3.00,3.00): 23 | (5.20,4.62): 3 | 0.0372 |
| No curriculum it100 | 63 | (2.00,1.00): 4, (4.00,4.00): 18, (1.00,1.00): 10, (3.00,3.00): 11, (2.00,2.00): 1 | (0.38,-0.20): 3, (5.20,4.62): 15, (4.62,5.20): 1 | 0.0253 |
| No multi-step SOCP it100 | 309 | (1.00,1.00): 148, (4.00,4.00): 47, (2.00,2.00): 28, (2.00,3.00): 5, (3.00,3.00): 66, (1.00,2.00): 1 | (4.62,5.20): 14 | 0.0421 |
| No progress it100 | 128 | (2.00,2.00): 3, (3.00,3.00): 36, (4.00,4.00): 27, (1.00,1.00): 56 | (5.20,4.62): 2, (4.62,5.20): 4 | 0.0306 |
