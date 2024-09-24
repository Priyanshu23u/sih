const { Rettiwt } = require("rettiwt-api");
const { apiKey } = require("./apiKey.js");
const fs = require('fs');
const path = require('path');

// Function to filter tweets and extract titles and links
const filterTweetsForIndiaDisasters = (tweets) => {
  const disasterKeywords = [
    "#earthquake",
    "#flood",
    "#hurricane",
    "#tornado",
    "#tsunami",
    "#volcano",
    "#cyclone",
    "#wildfire",
    "#landslide",
    "#avalanche",
    "#drought",
    "#heatwave",
    "#blizzard",
    "#storm",
    "#typhoon",
    "#hailstorm",
    "#mudslide",
    "#sandstorm",
    "#tremor",
    "#aftershock",
    "#flashflood",
    "#severeweather",
  ];

  return tweets
    .filter(
      (tweet) =>
        tweet.fullText.includes("#india") &&
        disasterKeywords.some((keyword) => tweet.fullText.includes(keyword))
    )
    .map(({ id, fullText }) => ({
      title: fullText,
      link: `https://twitter.com/user/status/${id}`, // Constructing tweet URL
    }));
};

// Function to append data to CSV
const appendToCSV = (data) => {
  const csvContent = data.map(({ title, link }) => `"${title.replace(/"/g, '""')}","${link}"`).join('\n');
  fs.appendFile(path.join(__dirname, 'news_data1.csv'), csvContent + '\n', (err) => {
    if (err) {
      console.error('Error appending to CSV:', err);
    } else {
      console.log('Data successfully appended to CSV');
    }
  });
};

// Function to search tweets and append titles and links to CSV
const testSearchTweets = (apiKey) => {
  const rettiwt = new Rettiwt({ apiKey }); // User authentication with API key

  rettiwt.tweet
    .search({
      hashtags: [
        "earthquake",
        "flood",
        "hurricane",
        "tornado",
        "tsunami",
        "volcano",
        "cyclone",
        "wildfire",
        "landslide",
        "avalanche",
        "drought",
        "heatwave",
        "blizzard",
        "storm",
        "typhoon",
        "hailstorm",
        "mudslide",
        "sandstorm",
        "tremor",
        "aftershock",
        "lightningstrike",
        "icestorm",
      ],
    })
    .then((data) => {
      const filteredTweets = filterTweetsForIndiaDisasters(data.list);
      appendToCSV(filteredTweets);
    })
    .catch((error) => {
      console.error("Error searching tweets:", error);
    });
};

// Call the function to test searching tweets and appending data
testSearchTweets(apiKey); // Replace with your actual API key
