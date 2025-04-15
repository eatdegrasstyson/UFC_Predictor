import csv
import math
from datetime import datetime

# Open and read the CSV file
with open("./177/ufc-master.csv", "r") as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)
    # for feature in header:
    #     print(feature)

# dict for fighter elos
fighters_elo = {}

# initializing dict with fighter names and base elo
base_elo = 1500
k_factor = 32   # how much rating changes after wins and losses
                # higher k factor because fighting is more random

# Function to calculate expected outcome
def expected_outcome(rating1, rating2):
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

# Function to update ELO ratings
def update_elo(winner_elo, loser_elo, k):
    expected_winner = expected_outcome(winner_elo, loser_elo)
    expected_loser = expected_outcome(loser_elo, winner_elo)
    
    # Update ELO ratings
    new_winner_elo = winner_elo + k * (1 - expected_winner)
    new_loser_elo = loser_elo + k * (0 - expected_loser)
    
    return new_winner_elo, new_loser_elo

# Optional: Function to adjust K-factor based on fight conditions
def adjust_k_factor(row):
    adjusted_k = k_factor
    
    # Title fights have higher impact
    if row["TitleBout"] == "TRUE":
        adjusted_k *= 1.2
    
    # Finish type affects impact (KO/TKO/SUB more decisive than decisions)
    if row["Finish"] in ["KO/TKO", "SUB"]:
        adjusted_k *= 1.1
    
    # TODO: Adjust k according to weight class since heavier classes are 
    #       more volatile. Also consider including betting odds to adjust
    #       less for one sided match ups (although I already do this a little)
    return adjusted_k

if __name__ == "__main__":
    # First pass: sort by date (if available)
    fights = []
    with open("./177/ufc-master.csv", "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            fights.append(row)
    
    fights.sort(key=lambda x: x["Date"] if x["Date"] else "")
    
    for row in fights:
        # Skip fights with no clear winner
        if not row["Winner"] or row["Winner"] not in ["Red", "Blue"]:
            continue
            
        win_side = row["Winner"]
        lose_side = "Blue" if win_side == "Red" else "Red"
    
        winner = row[win_side+"Fighter"]
        loser = row[lose_side+"Fighter"]
        
        # If fighters don't have an ELO rating yet, initialize them
        if winner not in fighters_elo:
            fighters_elo[winner] = base_elo
        if loser not in fighters_elo:
            fighters_elo[loser] = base_elo
        
        # Get current ELO ratings
        winner_elo = fighters_elo[winner]
        loser_elo = fighters_elo[loser]
        
        # Adjust K-factor based on fight conditions
        adjusted_k = adjust_k_factor(row)
        
        # Update ELO ratings
        new_winner_elo, new_loser_elo = update_elo(winner_elo, loser_elo, adjusted_k)
        
        # Update the ELO ratings dictionary
        fighters_elo[winner] = new_winner_elo
        fighters_elo[loser] = new_loser_elo
    
    # Save the ELO ratings to a CSV file
    with open("fighter_elo_ratings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fighter", "ELO"])
        
        # Sort by ELO rating (highest first)
        sorted_elos = sorted(fighters_elo.items(), key=lambda x: x[1], reverse=True)
        
        for fighter, elo in sorted_elos:
            writer.writerow([fighter, elo])
    
    # Optional: update the original CSV with ELO ratings at time of fight
    updated_fights = []
    # Reset fighter ELOs to recalculate them while saving pre-fight ELOs
    fighters_elo = {}
    
    for row in fights:
        if row["Winner"] in ["Red", "Blue"]:
            win_side = row["Winner"]
            lose_side = "Blue" if win_side == "Red" else "Red"
    
            winner = row[win_side+"Fighter"]
            loser = row[lose_side+"Fighter"]
            
            # If fighters don't have an ELO rating yet, initialize them
            if winner not in fighters_elo:
                fighters_elo[winner] = base_elo
            if loser not in fighters_elo:
                fighters_elo[loser] = base_elo
            
            # Store pre-fight ELOs
            row["RedElo"] = fighters_elo[row["RedFighter"]]
            row["BlueElo"] = fighters_elo[row["BlueFighter"]]
            row["EloDifference"] = fighters_elo[row["RedFighter"]] - fighters_elo[row["BlueFighter"]]
            
            # Update ELO ratings for next fight
            if row["Winner"] in ["Red", "Blue"]:
                adjusted_k = adjust_k_factor(row)
                new_winner_elo, new_loser_elo = update_elo(
                    fighters_elo[winner], 
                    fighters_elo[loser], 
                    adjusted_k
                )
                fighters_elo[winner] = new_winner_elo
                fighters_elo[loser] = new_loser_elo
        
        updated_fights.append(row)
    
    # Write the updated fights data with ELO information
    with open("ufc_fights_with_elo.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header + ["RedElo", "BlueElo", "EloDifference"])
        writer.writeheader()
        writer.writerows(updated_fights)
    
    print(f"ELO ratings calculated for {len(fighters_elo)} fighters.")
    print(f"Top 5 fighters by ELO rating:")
    for fighter, elo in sorted(fighters_elo.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{fighter}: {round(elo, 1)}")
