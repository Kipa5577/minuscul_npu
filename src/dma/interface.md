# dma — Réservé à la v2 (FPGA)

Ce bloc n'existe pas en v1 (D8) : l'hôte écrit directement dans les buffers.

À spécifier en v2 :
- interface maître vers la mémoire externe (Wishbone ou AXI) ;
- registres d'adresses source et destination ;
- handshake valid/ready aux frontières (D12), car la latence de la mémoire externe est variable ;
- éventuellement le double buffering (D6), pour recouvrir les transferts par le calcul.
