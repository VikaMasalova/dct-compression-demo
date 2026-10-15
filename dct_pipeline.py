"""
dct_pipeline.py
Аналогічний сценарій DCT-кодування/декодування з трохи зміненими повідомленнями.
Використання: python dct_pipeline.py --input 11.bmp
"""
import os
import argparse
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

NUM_BLOCKS_PREVIEW = 3

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def divide_into_blocks(img, block_size):
    h, w = img.shape
    blocks = []
    for r in range(0, h, block_size):
        row_blocks = []
        for c in range(0, w, block_size):
            blk = img[r:r+block_size, c:c+block_size]
            if blk.shape != (block_size, block_size):
                pad = np.zeros((block_size, block_size))
                pad[:blk.shape[0], :blk.shape[1]] = blk
                blk = pad
            row_blocks.append(blk)
        blocks.append(row_blocks)
    return blocks

def assemble_blocks(blocks, block_size, orig_shape):
    h, w = orig_shape
    out = np.zeros((h, w))
    for i, row in enumerate(blocks):
        for j, blk in enumerate(row):
            y1 = i * block_size
            x1 = j * block_size
            y2 = min(y1 + block_size, h)
            x2 = min(x1 + block_size, w)
            out[y1:y2, x1:x2] = blk[:(y2 - y1), :(x2 - x1)]
    return out

def main(args):
    # Пункт 1 — читання
    img_path = args.input
    img = Image.open(img_path).convert('L')
    orig = np.array(img, dtype=np.float64)
    print(f"КРОК 1: Файл '{img_path}' завантажено успішно.")

    # Пункт 2 — розміри й рівні яскравості
    h, w = orig.shape
    print(f"КРОК 2: Розмір зображення: {w} × {h} пікселів.")
    print(f"   Інтенсивність пікселів: від {orig.min():.0f} до {orig.max():.0f}")

    # Пункт 3 — показ оригіналу (збережемо картинку)
    plt.figure(figsize=(8, 10))
    plt.imshow(orig, cmap='gray')
    plt.title("КРОК 3 — Оригінал")
    plt.axis('off')
    os.makedirs(args.outdir, exist_ok=True)
    plt.savefig(os.path.join(args.outdir, "original.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Пункт 5 — розбиття на блоки
    N = args.block
    blocks = divide_into_blocks(orig, N)
    print(f"\nКРОК 5: Розбито на блоки розміру {N}×{N} — покажу перші {NUM_BLOCKS_PREVIEW}:")
    cnt = 0
    for i, row in enumerate(blocks):
        for j, b in enumerate(row):
            if cnt >= NUM_BLOCKS_PREVIEW:
                break
            print(f"\nБлок {cnt+1} (поз. {i},{j}):\n", b.astype(int))
            cnt += 1
        if cnt >= NUM_BLOCKS_PREVIEW:
            break
    total_blocks = len(blocks) * len(blocks[0])
    print(f"   Усього блоків: {len(blocks)} × {len(blocks[0])} = {total_blocks}")

    # Пункт 6 — DCT, зсув і квантування
    Mk = args.mk
    dct_blocks = []
    for row in blocks:
        rnew = []
        for b in row:
            shifted = b - 128.0
            coeffs = dct2(shifted)
            quant = np.round(coeffs / Mk)
            if len(dct_blocks) == 0 and len(rnew) < NUM_BLOCKS_PREVIEW:
                idx = len(rnew) + 1
                print(f"\nКРОК 6 — Блок {idx}:")
                print("  Вхідний блок:\n", b.astype(int))
                print("  DCT (округлено):\n", np.round(coeffs, 2))
                print("  Після квантування:\n", quant.astype(int))
            rnew.append(quant)
        dct_blocks.append(rnew)
    print(f"   Виконано DCT та квантування (Mk = {Mk}).")

    # Пункт 7 — фільтрація по діагоналі (i+j >= P -> 0)
    P = args.p
    filtered = []
    for row in dct_blocks:
        rnew = []
        for q in row:
            f = q.copy()
            for i in range(N):
                for j in range(N):
                    if i + j >= P:
                        f[i, j] = 0
            if len(filtered) == 0 and len(rnew) < NUM_BLOCKS_PREVIEW:
                idx = len(rnew) + 1
                print(f"\nКРОК 7 — Блок {idx} після обрізки (P={P}):")
                print(f.astype(int))
            rnew.append(f)
        filtered.append(rnew)
    print(f"   Коефіцієнти з i+j ≥ {P} обнулені.")

    # Пункт 8 — підрахунок Kc
    kept = P * (P + 1) // 2
    Kc = (N * N) / kept
    print(f"\nКРОК 8: Залишено {kept} коеф. з {N*N} → орієнтовне Kc = {Kc:.2f}")
    mask = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i + j < P:
                mask[i, j] = 1
    print("   Маска збережених коефіцієнтів (1 = зберігається):")
    print(mask)

    # Пункт 9 — відновлення (IDCT)
    restored_blocks = []
    for row in filtered:
        rnew = []
        for f in row:
            deq = f * Mk
            rec = idct2(deq) + 128.0
            rec = np.clip(rec, 0, 255)
            if len(restored_blocks) == 0 and len(rnew) < NUM_BLOCKS_PREVIEW:
                idx = len(rnew) + 1
                print(f"\nКРОК 9 — Відновлений блок {idx}:")
                print(np.round(rec).astype(int))
            rnew.append(rec)
        restored_blocks.append(rnew)

    restored_img = assemble_blocks(restored_blocks, N, orig.shape).astype(np.uint8)
    out_path = os.path.join(args.outdir, f"restored_Mk{Mk}_P{P}.png")
    Image.fromarray(restored_img).save(out_path)
    print(f"КРОК 9: Відновлене зображення збережено як: {out_path}")

    # Пункт 10 — параметричне дослідження (різні Mk при фіксованому P)
    os.makedirs(os.path.join(args.outdir, "results"), exist_ok=True)
    Mk_values = args.mk_values
    P_fixed = args.p_fixed
    kept_fixed = P_fixed * (P_fixed + 1) // 2
    Kc_fixed = (N*N) / kept_fixed
    print(f"\nКРОК 10: Варіювання Mk при P = {P_fixed} (Kc ≈ {Kc_fixed:.2f})")

    for Mkv in Mk_values:
        # повторюємо квантування/фільтрацію/відновлення
        dct_temp = []
        for row in blocks:
            rtemp = []
            for b in row:
                sb = b - 128.0
                d = dct2(sb)
                q = np.round(d / Mkv)
                rtemp.append(q)
            dct_temp.append(rtemp)

        filt_temp = []
        for row in dct_temp:
            rtemp = []
            for q in row:
                f = q.copy()
                for i in range(N):
                    for j in range(N):
                        if i + j >= P_fixed:
                            f[i, j] = 0
                rtemp.append(f)
            filt_temp.append(rtemp)

        rec_blocks = []
        for row in filt_temp:
            rtemp = []
            for f in row:
                dq = f * Mkv
                r = idct2(dq) + 128.0
                r = np.clip(r, 0, 255)
                rtemp.append(r)
            rec_blocks.append(rtemp)

        final_img = assemble_blocks(rec_blocks, N, orig.shape).astype(np.uint8)
        save_path = os.path.join(args.outdir, "results", f"Mk{Mkv}_P{P_fixed}.png")
        Image.fromarray(final_img).save(save_path)

        # збережена порівняльна картинка
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        ax1.imshow(orig, cmap='gray'); ax1.set_title("Оригінал"); ax1.axis('off')
        ax2.imshow(final_img, cmap='gray'); ax2.set_title(f"Mk={Mkv} | P={P_fixed} | Kc≈{Kc_fixed:.2f}"); ax2.axis('off')
        plt.suptitle(f"{img_path} — варіант Mk={Mkv}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"compare_Mk{Mkv}.png"), dpi=150)
        plt.close()

    print("КРОК 10: Дослідження завершено. Результати у папці 'results'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCT-based compression demo")
    parser.add_argument("--input", "-i", default="11.bmp", help="шлях до вхідного зображення")
    parser.add_argument("--block", "-b", type=int, default=8, help="розмір блоку N (за замовчуванням 8)")
    parser.add_argument("--mk", type=float, default=10.0, help="коефіцієнт масштабування Mk")
    parser.add_argument("--p", type=int, default=8, help="параметр P для селекції по діагоналі")
    parser.add_argument("--outdir", "-o", default="output", help="папка для збереження результатів")
    parser.add_argument("--mk-values", nargs='+', type=float, default=[1,5,10,15,20,30], help="список Mk для варіювання")
    parser.add_argument("--p-fixed", type=int, default=8, help="фіксоване P для пункту 10")
    args = parser.parse_args()
    main(args)
