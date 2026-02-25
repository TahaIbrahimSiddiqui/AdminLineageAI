import pandas as pd
from dotenv import load_dotenv
import json
from google import genai
from google.genai import types
import os

from ddlpy.utils import IEC, TMP

# Load environment variables from .env file
load_dotenv()


def get_llm_output(pc51_tehsil_list: list[str], pc01_subdistrict_list: list[str]):
    """
    Get LLM output for matching 1951 subdistricts to 2001 tehsils.
    """
    PROMPT = f"""
    System Role:
    You are a historically aware administrative-unit lineage engine specializing in Indian administrative history between 1950 and 2001.

    Goal:
    Map a list of "Old" subdistricts (existing in 1950) to their "New" descendant subdistricts (existing in 2001).

    Strict Requirements:
    True Lineage (1950 -> 2001):
    Map a NewSubdistrict to an OldSubdistrict only if the New one was administratively carved out of the Old one (e.g., via bifurcation or upgrading of a revenue circle) between 1950 and 2001.
    Do not link based on name similarity or geographic proximity alone; use historical administrative records.

    Scope & Filtering (Crucial):
    The New List is likely "noisy" (it may contain subdistricts from neighboring districts that are not relevant to the Old List).

    Filter Logic: If a NewSubdistrict descends from a parent that is not present in the provided Old List, exclude it entirely from the output. Do not force a match.

    Output Format:

    Output only a valid JSON object.
    Use the exact spelling/casing provided in the input lists.

    Structure: Group descendants under their 1950 parent key. You must include a reason field explaining the lineage (e.g., "Bifurcated in 1998").
    Required JSON Structure:
    JSON
    {{
        "Name_From_Old_List_A": {{
            "descendants": [
            "Name_From_New_List_1",
            "Name_From_New_List_2"
            ],
            "reason": "Historical reason for the split or name change."
    }},
        "Name_From_Old_List_B": {{
            "descendants": [
            "Name_From_New_List_3"
            ],
            "reason": "Boundaries remained largely unchanged between 1950 and 2001."
        }}
    }}

    Input Data:
    Old Subdistricts (1950):
    {json.dumps(pc51_tehsil_list)}
    New Subdistricts (2001):
    {json.dumps(pc01_subdistrict_list)}
    """

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = "gemini-3-pro-preview"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=PROMPT),
            ],
        ),
    ]
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        tools=tools,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    try:
        llm_output = response.text
        llm_output_json = json.loads(llm_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM output as JSON: {e}\nOutput was: {llm_output}"
        )
    return llm_output_json


def read_tehsil_llm_output_json(pc51_state_name: str, pc51_district_name: str):
    """
    Read LLM output JSON for a given state and district.
    """
    file_path = (
        TMP
        / "pc51"
        / "tehsil_llm_output"
        / f"{pc51_state_name}_{pc51_district_name}_llm_output.json"
    )
    if not file_path.exists():
        return None
    with open(file_path, "r") as f:
        llm_output_json = json.load(f)
    return llm_output_json


if __name__ == "__main__":
    # Read PC51 cleaned data
    pc51_cleaned_df = pd.read_stata(TMP / "pc51/pc51_cleaned.dta")

    # Read PC51-01 subdistrict cleaned list
    pc51_01_subdistrict_clean = pd.read_stata(
        TMP / "pc51/pc51_01_subdistrict_clean.dta"
    )

    # Only keep relevant columns
    cols_to_keep = ["pc51_state_name", "pc51_district_name", "tehsil_clean"]
    pc51_cleaned_df = pc51_cleaned_df[cols_to_keep]

    # Drop duplicates
    pc51_cleaned_df = pc51_cleaned_df.drop_duplicates()

    # Get unique district list
    pc51_district_list = (
        pc51_cleaned_df[["pc51_state_name", "pc51_district_name"]]
        .drop_duplicates()
        .values.tolist()
    )

    mapping_rows = []

    for state, district in pc51_district_list:
        print(f"Processing district: {district}")
        # Get PC51 tehsil list for the district
        pc51_tehsil_list = (
            pc51_cleaned_df[
                (pc51_cleaned_df["pc51_state_name"] == state)
                & (pc51_cleaned_df["pc51_district_name"] == district)
            ]["tehsil_clean"]
            .unique()
            .tolist()
        )

        # Remove empty tehsils
        pc51_tehsil_list = [
            tehsil for tehsil in pc51_tehsil_list if tehsil.strip() != ""
        ]
        print(pc51_tehsil_list)

        # Get PC01 subdistrict list for the district
        pc01_subdistrict_list = (
            pc51_01_subdistrict_clean[
                (pc51_01_subdistrict_clean["pc51_state_name"] == state)
                & (pc51_01_subdistrict_clean["pc51_district_name"] == district)
            ]["pc01_subdistrict_name"]
            .unique()
            .tolist()
        )
        # Remove empty subdistricts
        pc01_subdistrict_list = [
            subdistrict
            for subdistrict in pc01_subdistrict_list
            if subdistrict.strip() != ""
        ]
        print(pc01_subdistrict_list)

        if len(pc01_subdistrict_list) == 0:
            print(f"No PC01 subdistricts found for state: {state} district: {district}")
            continue

        llm_output_json = read_tehsil_llm_output_json(state, district)
        if llm_output_json is None:
            # Get LLM output
            try:
                llm_output_json = get_llm_output(
                    pc51_tehsil_list, pc01_subdistrict_list
                )
            except ValueError as e:
                print(f"Error processing state: {state} district: {district}: {e}")
                continue
            # Save LLM output to file
            output_file_path = (
                TMP
                / "pc51"
                / "tehsil_llm_output"
                / f"{state}_{district}_llm_output.json"
            )
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w") as f:
                json.dump(llm_output_json, f, indent=4)

        # Create mapping rows
        for pc51_tehsil_name, details in llm_output_json.items():
            descendants = details.get("descendants", [])
            for pc01_subdistrict_name in descendants:
                mapping_rows.append(
                    {
                        "pc51_state_name": state,
                        "pc51_district_name": district,
                        "tehsil_clean": pc51_tehsil_name,
                        "pc01_subdistrict_name": pc01_subdistrict_name,
                    }
                )

    # Create mapping DataFrame
    mapping_df = pd.DataFrame(mapping_rows)
    # Save mapping DataFrame to Stata file
    mapping_df.to_stata(
        IEC / "pc51/merge/pc51_tehsil_pc01_subdistrict_key.dta",
        write_index=False,
    )
    print("Mapping file saved to $iec/pc51/merge/pc51_tehsil_pc01_subdistrict_key.dta")
