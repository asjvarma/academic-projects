install.packages(c("readxl"))
library(readxl)

data <- read_excel("Downloads/LCA_Disclosure_Data_FY2023_Q1.xlsx")

# Extract the top 5 states with the most H1B visa cases
top_states <- data %>%
  filter(VISA_CLASS == "H1B") %>%
  group_by(EMPLOYER_STATE) %>%
  summarise(total_cases = n()) %>%
  top_n(5, total_cases)

# Create the bar graph and save it as a PNG file
png("top_5_employer_states_h1b_cases_bar.png", width = 800, height = 600)
ggplot(top_states, aes(x = EMPLOYER_STATE, y = total_cases, fill = EMPLOYER_STATE)) +
  geom_bar(stat = "identity", color = "black", width = 0.5) +
  ggtitle("Top 5 Employer States with Most H1B Visa Cases") +
  xlab("Employer State") +
  ylab("Total H1B Visa Cases") +
  theme_bw()
dev.off()