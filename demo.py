import argparse, os
from human_action_predictor import HumanActionPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_model4.pth")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", choices=["cpu","cuda"], default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Image not found: {args.image}")
        return

    predictor = HumanActionPredictor(args.model, device=args.device)
    action, confidence = predictor.predict(args.image)
    print(f"Action: {action}\nConfidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
