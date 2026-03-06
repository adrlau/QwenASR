{
  description = "QwenASR - Pure Rust CPU-only inference engine for Qwen3-ASR speech-to-text";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        qwen-asr = pkgs.rustPlatform.buildRustPackage {
          pname = "qwen-asr";
          version = "0.3.1";

          src = ./.;

          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = [ pkgs.pkg-config ];

          buildInputs = [
            pkgs.openblas
            pkgs.alsa-lib
          ];

          # Build only the CLI binary
          cargoBuildFlags = [
            "--package"
            "qwen-asr-cli"
          ];

          # Tests require a downloaded model and sample audio files
          doCheck = false;

          env = {
            RUSTFLAGS = "-C target-cpu=native";
            OPENBLAS_DIR = "${pkgs.openblas}";
          };

          meta = with pkgs.lib; {
            description = "Pure Rust CPU-only inference engine for Qwen3-ASR speech-to-text models";
            homepage = "https://github.com/adrlau/QwenASR";
            license = licenses.mit;
            mainProgram = "qwen-asr";
            platforms = platforms.linux;
          };
        };
      in
      {
        packages = {
          default = qwen-asr;
          inherit qwen-asr;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ qwen-asr ];

          packages = with pkgs; [
            cargo
            rustc
            rust-analyzer
            clippy
            rustfmt
            qwen-asr
          ];

          env = {
            RUSTFLAGS = "-C target-cpu=native";
            OPENBLAS_DIR = "${pkgs.openblas}";
          };
        };
      }
    );
}
