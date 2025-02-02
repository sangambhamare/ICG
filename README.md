# BLIP Image Captioning App

This repository contains a Streamlit application that leverages the BLIP model for image captioning. The app allows users to either upload an image or provide an image URL, and then generate creative image captions using two modes:

- **Conditional Captioning:** Generate captions based on a provided text prompt.
- **Unconditional Captioning:** Generate captions without any prompt.

The BLIP model (from [Salesforce](https://www.salesforce.com/)) is integrated via the Hugging Face Transformers library.

## Features

- **User-Friendly Interface:** Easily upload images or enter URLs.
- **Two Captioning Modes:** Choose between conditional and unconditional caption generation.
- **Quick Setup and Deployment:** Run locally or deploy on Streamlit Cloud.
- **Minimal Dependencies:** Uses Streamlit, Transformers, Pillow, and Requests.

## Requirements

- Python 3.7+
- See [requirements.txt](requirements.txt) for the full list of dependencies.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
