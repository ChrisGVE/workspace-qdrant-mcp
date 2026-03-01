defmodule Bookshelf.MixProject do
  use Mix.Project

  def project do
    [
      app: :bookshelf,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: false,
      deps: []
    ]
  end

  def application do
    [extra_applications: []]
  end
end
