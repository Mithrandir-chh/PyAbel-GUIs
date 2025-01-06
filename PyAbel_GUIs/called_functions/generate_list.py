def generate_ranges(ranges, step=3):
    result = []

    if step == 0:
        # Generate point ranges where start = end
        for start_range, end_range in ranges:
            current = start_range
            while current <= end_range:
                result.append((current, current))
                current += 1
    else:
        # Original behavior for step > 0
        for start_range, end_range in ranges:
            current = start_range
            while current < end_range:
                result.append((current, current + step))
                current += step

    return result

def generate_rolling_ranges(ranges, rolling_step=3):
    result = []
    for start_range, end_range in ranges:
        current = start_range
        while current < end_range:
            result.append((current, current + rolling_step))
            current += 1
    return result

delta = 140 # make delta even if step is odd, make delta odd if step is even
broad_delta = 40
# peak1 = 138
peak2 = 200
# peak3 = 99
# peak3 = 380
# broadpeak = type in the peak
input_ranges = [
    # (peak1-delta, peak1+delta),  # First group of ranges
    (peak2-delta, peak2+delta),  # Second group of ranges
    # (peak3-delta, peak3+delta)  # Third group of ranges
]

# r_range = generate_ranges(input_ranges, step=0)
r_range = generate_rolling_ranges(input_ranges, rolling_step=5)

# Print the ranges in a formatted way
print("r_range = [", end="")
for i, (start, end) in enumerate(r_range):
    if i % 5 == 0 and i != 0:  # Start a new line every 5 items
        print()
    print(f"({start}, {end})", end="")
    if i != len(r_range) - 1:
        print(", ", end="")
print("]")