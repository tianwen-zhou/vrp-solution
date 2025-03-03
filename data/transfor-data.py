import csv

# Define the input and output file paths
input_csv = 'data_jilin.csv'  # Replace with the path to your input CSV file
output_csv = 'waybills_data.csv'  # Path to the output CSV file

# Open the input CSV file and read the data
with open(input_csv, mode='r') as infile:
    reader = csv.DictReader(infile)

    # Define the output CSV fieldnames
    header = [
        "WaybillNumber", "Sender", "SenderAddress", "SenderPhone",
        "Recipient", "RecipientAddress", "RecipientPhone",
        "PostCode", "RouteCode", "ParcelValue", "DeliveryTime",
        "SenderLatitude", "SenderLongitude", "RecipientLatitude", "RecipientLongitude", "Status"
    ]

    # Open the output CSV file and write the transformed data
    with open(output_csv, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header)

        # Write the header row
        writer.writeheader()

        # Loop through the input data and transform it to match the new format
        for row in reader:
            transformed_row = {
                "WaybillNumber": row["order_id"],
                "Sender": "UnknownSender",  # Placeholder for sender
                "SenderAddress": row["city"],  # Use city for sender address
                "SenderPhone": "Unknown",  # Placeholder for sender phone
                "Recipient": row["courier_id"],  # Use courier_id as recipient
                "RecipientAddress": row["region_id"],  # Use region_id as recipient address
                "RecipientPhone": "Unknown",  # Placeholder for recipient phone
                "PostCode": row["region_id"],  # Use region_id for postcode
                "RouteCode": row["aoi_id"],  # Use aoi_id as route code
                "ParcelValue": "0",  # Placeholder for parcel value
                "DeliveryTime": row["delivery_time"],  # Use delivery_time as is
                "SenderLatitude": row["lat"],  # Use lat for sender latitude
                "SenderLongitude": row["lng"],  # Use lng for sender longitude
                "RecipientLatitude": row["delivery_gps_lat"],  # Use delivery_gps_lat for recipient latitude
                "RecipientLongitude": row["delivery_gps_lng"],  # Use delivery_gps_lng for recipient longitude
                "Status": "1"  # Placeholder for status
            }

            # Write the transformed row to the output file
            writer.writerow(transformed_row)

print(f"CSV file '{output_csv}' has been generated successfully.")
