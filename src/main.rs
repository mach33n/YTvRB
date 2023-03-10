//#![feature(core_intrinsics)]
//use ::std::intrinsics::breakpoint;
use hyper::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

    let api_key: &str = "AIzaSyDUQyRdH2MOj-JG0bjLrydbNVC7-EjzCqo";
    let client = Client::new();

    let uri: &str = "https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet%2Creplies";
    
    //Generate Video IDs
    

    //Iterate over Video IDs
    let video_ids = ["_VB39Jo8mAQ", "aX7jnVXXG5o"];
    for id in video_ids {
        let temp_uri: hyper::Uri = format!("{}&videoId={}&key={}", uri, id, api_key).parse()?;
        println!("{}", temp_uri);
        let res = client.get(temp_uri).await?;

        println!("Response Status: {}", res.status());

        let buf = hyper::body::to_bytes(res).await?;
        println!("Response Body: {:?}", buf);
    }
    Ok(())
}
