  {
  description = "Description for the project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    custom-nixpkgs.url = "github:Open-Systems-Innovation/custom-nixpkgs";
  };

  outputs = { self, nixpkgs, custom-nixpkgs, ... }:
      let
        system = "x86_64-linux";
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ custom-nixpkgs.overlays.default ];
          config.allowUnfree = true;
        };
      in
        {
          devShells.${system}.default = pkgs.mkShell {
            name = "default";
               
            packages = [
            # General packages
              # pkgs.hello-nix
              # pkgs.petsc
              # pkgs.mpich
              # pkgs.clangd
              #  # Python packages
              (pkgs.python312.withPackages (python-pkgs: [
              #  # packages for formatting/ IDE
              #  python-pkgs.pip
                python-pkgs.python-lsp-server
                python-pkgs.pep8
                python-pkgs.flake8
              #  # packages for code
                python-pkgs.imageio
                python-pkgs.ipython
                python-pkgs.matplotlib
                python-pkgs.numpy
                python-pkgs.pyvista
                python-pkgs.scipy
                python-pkgs.torch
              ]))
            ];

            # PETSC_DIR = "${pkgs.petsc}";

            shellHook = ''
              export VIRTUAL_ENV="Custom Environment"
            '';
          };
        };
}

