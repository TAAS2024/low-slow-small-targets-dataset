"""
Content Verifier — CLIP/BLIP-based image verification for generated scenes.
- Background verification: does generated bg match expected scene semantics?
- Drone detection: is there a drone visible in the final image?
- Reference comparison: compare against real Anti-UAV background images.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reference background images (Anti-UAV, drone-free)
VAULT_ROOT = Path(__file__).parent.parent
REF_BG_DIR = VAULT_ROOT / "1-background-pool" / "RGB_raw_frames_antiuav"


class ContentVerifier:
    """
    CLIP-based content verification for generated scenes.

    Usage:
        verifier = ContentVerifier()
        score = verifier.verify_background(image, "industrial factory, smokestacks")
        drone_score = verifier.detect_drone(image)
        ref_score = verifier.compare_to_reference(image, n_samples=50)
    """

    def __init__(self, device="cuda"):
        self.device = device
        self._model = None
        self._processor = None
        self._ref_embeddings = None
        self._loaded = False

    def load_models(self):
        if self._loaded:
            return

        logger.info("Loading CLIP ViT-B/32...")
        from transformers import CLIPProcessor, CLIPModel

        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model.eval()
        self._loaded = True
        logger.info("CLIP loaded.")

    def _encode_image(self, image):
        """Encode a PIL Image to CLIP embedding."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self._model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def _encode_text(self, text):
        """Encode text to CLIP embedding."""
        inputs = self._processor(text=text, return_tensors="pt", padding=True,
                                 truncation=True).to(self.device)
        with torch.no_grad():
            emb = self._model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    # ── 1. Background Semantic Verification ─────────────────────────

    def verify_background(self, image, expected_scene, negative_scenes=None):
        """
        Verify background matches expected scene semantics.

        Args:
            image: PIL Image or path
            expected_scene: expected scene description (e.g. "industrial factory")
            negative_scenes: list of descriptions that should NOT match

        Returns:
            dict with:
                - score: cosine similarity with expected scene
                - negative_scores: similarities with each negative scene
                - pass: True if score > all negative scores and score > 0.25
        """
        if not self._loaded:
            self.load_models()

        img_emb = self._encode_image(image)

        expected_emb = self._encode_text(expected_scene)
        score = float((img_emb @ expected_emb.T).item())

        result = {"score": score, "pass": score > 0.25}

        if negative_scenes:
            neg_embs = self._encode_text(negative_scenes)
            neg_scores = (img_emb @ neg_embs.T).squeeze().tolist()
            if isinstance(neg_scores, float):
                neg_scores = [neg_scores]
            result["negative_scores"] = dict(zip(negative_scenes, neg_scores))
            result["pass"] = result["pass"] and all(score > ns + 0.02 for ns in neg_scores)

        return result

    # ── 2. Drone Detection ─────────────────────────────────────────

    def detect_drone(self, image):
        """
        Check if a drone is present in the image using CLIP zero-shot.

        Returns:
            dict with drone_score, and comparison against "no drone" baseline
        """
        if not self._loaded:
            self.load_models()

        img_emb = self._encode_image(image)

        drone_texts = [
            "a photo of a quadcopter drone flying in the sky",
            "an unmanned aerial vehicle in the air",
            "a small drone with four rotors",
        ]
        no_drone_texts = [
            "a photo with no drones, just buildings and sky",
            "an empty sky with no flying objects",
            "industrial landscape without any aircraft",
        ]

        drone_emb = self._encode_text(drone_texts)
        no_drone_emb = self._encode_text(no_drone_texts)

        drone_score = float((img_emb @ drone_emb.T).mean().item())
        no_drone_score = float((img_emb @ no_drone_emb.T).mean().item())

        return {
            "drone_score": drone_score,
            "no_drone_score": no_drone_score,
            "drone_detected": drone_score > no_drone_score + 0.02,
        }

    # ── 3. Reference Comparison ─────────────────────────────────────

    def load_reference_embeddings(self, ref_dir=None, n_samples=100, cache=True):
        """
        Pre-compute CLIP embeddings for reference background images.

        Args:
            ref_dir: path to reference images (default: Anti-UAV backgrounds)
            n_samples: max number of reference images to load
            cache: if True, reuse cached embeddings
        """
        ref_dir = Path(ref_dir or REF_BG_DIR)

        cache_path = Path(__file__).parent / ".ref_embeddings.npy"
        if cache and cache_path.exists():
            logger.info(f"Loading cached reference embeddings ({n_samples} samples)...")
            self._ref_embeddings = np.load(cache_path)
            if len(self._ref_embeddings) > n_samples:
                self._ref_embeddings = self._ref_embeddings[:n_samples]
            return

        if not self._loaded:
            self.load_models()

        image_paths = sorted(ref_dir.glob("*.jpg")) + sorted(ref_dir.glob("*.png"))
        if not image_paths:
            image_paths = sorted(ref_dir.glob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        image_paths = image_paths[:n_samples]

        logger.info(f"Computing embeddings for {len(image_paths)} reference images...")
        embeddings = []
        for p in image_paths:
            try:
                emb = self._encode_image(p)
                embeddings.append(emb.cpu().numpy())
            except Exception as e:
                logger.warning(f"  skip {p.name}: {e}")

        self._ref_embeddings = np.concatenate(embeddings, axis=0)

        if cache:
            np.save(cache_path, self._ref_embeddings)
            logger.info(f"Saved {len(self._ref_embeddings)} embeddings to {cache_path}")

    def compare_to_reference(self, image, top_k=5):
        """
        Compare generated image against reference background images.
        Returns CLIP similarity distribution (mean, std, top-k).

        Args:
            image: PIL Image or path
            top_k: number of closest matches to return

        Returns:
            dict with mean_sim, std_sim, max_sim, top_k_matches
        """
        if not self._loaded:
            self.load_models()

        if self._ref_embeddings is None:
            self.load_reference_embeddings()

        img_emb = self._encode_image(image).cpu().numpy()

        similarities = (img_emb @ self._ref_embeddings.T).squeeze()

        return {
            "mean_sim": float(similarities.mean()),
            "std_sim": float(similarities.std()),
            "max_sim": float(similarities.max()),
            "min_sim": float(similarities.min()),
            "top_k": float(np.sort(similarities)[-top_k:][::-1].tolist()) if isinstance(similarities, np.ndarray) else [],
            "pass": float(similarities.mean()) > 0.20,
        }

    # ── 4. Full Verification Pipeline ───────────────────────────────

    def verify_full_pipeline(self, step1_bg, step3_final, expected_scene,
                             negative_scenes=None):
        """
        Run full verification on all pipeline outputs.

        Returns dict:
            - bg_verified: background semantic check
            - drone_detected: drone presence check on final image
            - ref_comparison: similarity to real backgrounds
        """
        results = {}

        logger.info("Verifying background semantics...")
        results["bg_verified"] = self.verify_background(
            step1_bg, expected_scene, negative_scenes)

        logger.info("Detecting drone in final image...")
        results["drone_detected"] = self.detect_drone(step3_final)

        logger.info("Comparing against reference backgrounds...")
        results["ref_comparison"] = self.compare_to_reference(step1_bg)

        # Summary
        all_pass = (
            results["bg_verified"]["pass"] and
            results["drone_detected"]["drone_detected"] and
            results["ref_comparison"]["pass"]
        )
        results["all_pass"] = all_pass

        return results

    def unload(self):
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        self._loaded = False
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Quick test
    verifier = ContentVerifier()

    test_img = "outputs/step1_background.png"
    if Path(test_img).exists():
        r = verifier.verify_background(
            test_img,
            "industrial factory with smokestacks and cooling towers",
            ["forest with trees", "ocean beach", "city street"],
        )
        print(f"BG verify: {r}")

        r2 = verifier.detect_drone("outputs/final_scene_000.png")
        print(f"Drone detect: {r2}")

    verifier.unload()
